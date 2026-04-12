"""
Writer-adversarial training components for writer-independent HTR.

Implements a Gradient Reversal Layer (GRL) and a WriterAdversarialHead
that encourages the feature extractor to learn writer-invariant representations
by jointly minimising the recognition loss and maximising writer confusion.

Reference:
    Ganin & Lempitsky, "Unsupervised Domain Adaptation by Backpropagation",
    ICML 2015.
"""

from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

try:
    import paddle
    import paddle.nn as nn
    from paddle.autograd import PyLayer

    class _GradientReversalFunction(PyLayer):
        """
        Custom autograd function that reverses the gradient during backprop.

        Forward pass: identity (returns input unchanged).
        Backward pass: negates the gradient, scaled by lambda_.
        """

        @staticmethod
        def forward(ctx, x: paddle.Tensor, lambda_: float) -> paddle.Tensor:  # type: ignore[override]
            ctx.save_for_backward(paddle.to_tensor(lambda_))
            return x.clone()

        @staticmethod
        def backward(ctx, grad_output: paddle.Tensor):  # type: ignore[override]
            (lambda_,) = ctx.saved_tensor()
            lambda_val = float(lambda_.numpy())
            return -lambda_val * grad_output, None

    class GradientReversalLayer(nn.Layer):
        """
        Gradient Reversal Layer (GRL).

        Passes activations through unchanged during the forward pass, but
        reverses (negates and scales by lambda_) the gradients during
        backpropagation.  Typically placed between the shared feature
        extractor and the writer discriminator head.

        Args:
            lambda_: Gradient reversal strength (lambda >= 0).  Can be
                     updated dynamically during training (e.g. annealed from
                     0 to 1 following the schedule in Ganin & Lempitsky).
        """

        def __init__(self, lambda_: float = 1.0) -> None:
            super().__init__()
            self.lambda_ = lambda_

        def forward(self, x: paddle.Tensor) -> paddle.Tensor:
            return _GradientReversalFunction.apply(x, self.lambda_)

        def set_lambda(self, lambda_: float) -> None:
            """Update the reversal strength (call during training loop)."""
            self.lambda_ = lambda_

        def extra_repr(self) -> str:
            return f"lambda_={self.lambda_:.4f}"

    # ---------------------------------------------------------------------------
    # Writer Adversarial Head
    # ---------------------------------------------------------------------------

    class WriterAdversarialHead(nn.Layer):
        """
        Writer discriminator head with gradient reversal.

        Architecture:
            GRL  ->  Linear(in_features, hidden_dim)  ->  ReLU  ->
            Dropout  ->  Linear(hidden_dim, n_writers)  ->  LogSoftmax

        The GRL ensures that gradients flowing back into the shared feature
        extractor are reversed, pushing it to produce writer-invariant features.

        Args:
            in_features:  Number of input feature dimensions (from backbone).
            n_writers:    Number of writer classes in the training set.
            hidden_dim:   Size of the intermediate discriminator layer.
            dropout_p:    Dropout probability.
            lambda_:      Initial GRL reversal strength.
        """

        def __init__(
            self,
            in_features: int,
            n_writers: int,
            hidden_dim: int = 256,
            dropout_p: float = 0.3,
            lambda_: float = 1.0,
        ) -> None:
            super().__init__()

            self.grl = GradientReversalLayer(lambda_=lambda_)

            self.discriminator = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_p),
                nn.Linear(hidden_dim, n_writers),
            )

            self.log_softmax = nn.LogSoftmax(axis=-1)

        def forward(self, features: paddle.Tensor) -> paddle.Tensor:
            """
            Forward pass.

            Args:
                features: Feature tensor of shape (B, in_features).

            Returns:
                Log-probability tensor of shape (B, n_writers).
            """
            reversed_features = self.grl(features)
            logits = self.discriminator(reversed_features)
            return self.log_softmax(logits)

        def set_lambda(self, lambda_: float) -> None:
            """Propagate a new GRL lambda to the reversal layer."""
            self.grl.set_lambda(lambda_)

        def compute_loss(
            self,
            features: paddle.Tensor,
            writer_labels: paddle.Tensor,
        ) -> paddle.Tensor:
            """
            Compute NLL loss for writer discriminator.

            Args:
                features:      Feature tensor (B, in_features).
                writer_labels: Integer writer-id labels (B,).

            Returns:
                Scalar loss tensor.
            """
            log_probs = self.forward(features)
            loss_fn = nn.NLLLoss()
            return loss_fn(log_probs, writer_labels)

        @staticmethod
        def anneal_lambda(
            current_step: int,
            total_steps: int,
            gamma: float = 10.0,
            lambda_max: float = 1.0,
        ) -> float:
            """
            Compute the annealed GRL lambda using the schedule from Ganin &
            Lempitsky (2015):

                lambda = lambda_max * (2 / (1 + exp(-gamma * p)) - 1)

            where p = current_step / total_steps.

            Args:
                current_step: Current training step (0-indexed).
                total_steps:  Total number of training steps.
                gamma:        Annealing rate (higher = faster ramp-up).
                lambda_max:   Maximum lambda value.

            Returns:
                Annealed lambda as a float.
            """
            p = current_step / max(total_steps, 1)
            return float(lambda_max * (2.0 / (1.0 + np.exp(-gamma * p)) - 1.0))

except ImportError:
    # paddle not installed — provide stub classes so the module can be imported
    # in environments without PaddlePaddle for testing/linting purposes.

    import warnings

    warnings.warn(
        "PaddlePaddle is not installed. GradientReversalLayer and "
        "WriterAdversarialHead are not available.",
        ImportWarning,
        stacklevel=2,
    )

    class GradientReversalLayer:  # type: ignore[no-redef]
        """Stub: PaddlePaddle not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError("PaddlePaddle is required. Install paddlepaddle or paddlepaddle-gpu.")

    class WriterAdversarialHead:  # type: ignore[no-redef]
        """Stub: PaddlePaddle not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError("PaddlePaddle is required. Install paddlepaddle or paddlepaddle-gpu.")


# ---------------------------------------------------------------------------
# CLI / __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        import paddle

        print("PaddlePaddle version:", paddle.__version__)

        B, D, W = 4, 512, 8  # batch, feature dim, n_writers
        features = paddle.randn([B, D])
        writer_ids = paddle.randint(0, W, shape=[B])

        # Test GRL
        grl = GradientReversalLayer(lambda_=1.0)
        out = grl(features)
        assert out.shape == features.shape, "GRL output shape mismatch"
        print(f"GRL forward OK: input={features.shape}, output={out.shape}")

        # Test WriterAdversarialHead
        head = WriterAdversarialHead(in_features=D, n_writers=W, hidden_dim=128)
        log_probs = head(features)
        assert log_probs.shape == [B, W], f"Expected ({B},{W}), got {log_probs.shape}"
        print(f"WriterAdversarialHead forward OK: log_probs shape={log_probs.shape}")

        loss = head.compute_loss(features, writer_ids)
        print(f"Adversarial loss: {loss.numpy().item():.4f}")

        # Annealing demo
        print("\nLambda annealing schedule (first 10 steps of 100):")
        for step in range(0, 100, 10):
            lam = WriterAdversarialHead.anneal_lambda(step, total_steps=100)
            print(f"  step={step:3d}  lambda={lam:.4f}")

    except ImportError as e:
        print(f"Cannot run demo: {e}")
