import numpy as np
import os
from tqdm import tqdm

class RunnerM():
    """
    This is an exmaple to train, evaluate, save, load the model. However, some of the function calling may not be correct 
    due to the different implementation of those models.
    """
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []
        '''
        # Deprecated reference: older checkpoints / notebooks assumed only train + dev curves existed.
        # (``test_scores`` / ``test_loss`` are now optional parallel series when ``test_set`` is passed to train.)
        '''
        # English: Optional per-iteration metrics on the held-out **test** set (same length as train_* when used).
        self.test_scores = []
        self.test_loss = []

    def train(self, train_set, dev_set, **kwargs):

        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        save_dir = kwargs.get("save_dir", "best_model")
        # eval_every: run full dev-set evaluation only every N training iterations (default 1 = old behavior).
        # Larger N greatly speeds up CNN training because evaluate() forwards the entire dev set at once.
        eval_every = max(1, int(kwargs.get("eval_every", 1)))
        # dev_batch_size: if set, evaluate() runs several smaller forwards (still full dev coverage); lowers peak RAM
        # and can improve throughput when NumPy is linked to multi-threaded BLAS (OpenBLAS/MKL).
        dev_batch_size = kwargs.get("dev_batch_size", None)
        if dev_batch_size is not None:
            dev_batch_size = max(1, int(dev_batch_size))

        '''
        # Deprecated: single-level ``os.mkdir`` failed when ``save_dir`` had missing parent folders.
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        '''
        # English: Ensure checkpoint directory exists (nested paths, e.g. codes/best_models/<relpath>/).
        os.makedirs(save_dir, exist_ok=True)

        # English: Optional ``(X_test, y_test)`` evaluated on the same cadence as validation (see ``eval_every``).
        test_set = kwargs.get("test_set", None)

        best_score = 0
        # Last dev metrics; reused on skipped eval steps so list lengths match train iterations for plotting.
        dev_score, dev_loss = 0.0, 0.0
        '''
        # Deprecated: test-set scalars were not tracked each step (only dev_score / dev_loss existed).
        '''
        # English: Last test metrics; reused on skipped eval steps when ``test_set`` is provided.
        test_score, test_loss = 0.0, 0.0

        # added setting: early stopping
        early_stop = bool(kwargs.get("early_stop", False))
        early_stop_patience = max(1, int(kwargs.get("early_stop_patience", 5)))
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            # added variable: best score at beginning of epoch
            best_at_epoch_start = best_score
            X, y = train_set

            assert X.shape[0] == y.shape[0]

            idx = np.random.permutation(range(X.shape[0]))

            X = X[idx]
            y = y[idx]

            # --- Mini-batch slicing (replaced loop) ---
            # English note: the previous loop used int(n / batch_size) + 1 iterations. When the
            # dataset size is an exact multiple of batch_size, that extra iteration produced an
            # empty slice (0 examples), then forward/backward/step still ran and corrupted state.
            # We now use ceiling division so the last batch is partial when needed and never empty.
            '''
            for iteration in range(int(X.shape[0] / self.batch_size) + 1):
                train_X = X[iteration * self.batch_size : (iteration+1) * self.batch_size]
                train_y = y[iteration * self.batch_size : (iteration+1) * self.batch_size]
            '''
            n_samples = X.shape[0]
            num_batches = (n_samples + self.batch_size - 1) // self.batch_size
            for iteration in range(num_batches):
                start = iteration * self.batch_size
                end = min(start + self.batch_size, n_samples)
                train_X = X[start:end]
                train_y = y[start:end]

                logits = self.model(train_X)
                trn_loss = self.loss_fn(logits, train_y)
                self.train_loss.append(trn_loss)
                
                trn_score = self.metric(logits, train_y)
                self.train_scores.append(trn_score)

                # the loss_fn layer will propagate the gradients.
                self.loss_fn.backward()

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                
                '''
                # Deprecated: refresh **validation** only on ``eval_every`` (no optional test-set logging).
                if iteration % eval_every == 0:
                    dev_score, dev_loss = self.evaluate(dev_set, dev_batch_size=dev_batch_size)
                self.dev_scores.append(dev_score)
                self.dev_loss.append(dev_loss)
                '''
                # English: Recompute dev (and optionally test) every ``eval_every`` steps; otherwise reuse last values.
                if iteration % eval_every == 0:
                    dev_score, dev_loss = self.evaluate(dev_set, dev_batch_size=dev_batch_size)
                    if test_set is not None:
                        test_score, test_loss = self.evaluate(test_set, dev_batch_size=dev_batch_size)
                self.dev_scores.append(dev_score)
                self.dev_loss.append(dev_loss)
                if test_set is not None:
                    self.test_scores.append(test_score)
                    self.test_loss.append(test_loss)

                if (iteration) % log_iters == 0:
                    print(f"epoch: {epoch}, iteration: {iteration}")
                    print(f"[Train] loss: {trn_loss}, score: {trn_score}")
                    print(f"[Dev] loss: {dev_loss}, score: {dev_score}")

            # If we skipped dev eval within the epoch, refresh once here so best checkpoint uses up-to-date metrics.
            if eval_every > 1:
                dev_score, dev_loss = self.evaluate(dev_set, dev_batch_size=dev_batch_size)
                if test_set is not None:
                    test_score, test_loss = self.evaluate(test_set, dev_batch_size=dev_batch_size)

            if dev_score > best_score:
                save_path = os.path.join(save_dir, 'best_model.pickle')
                self.save_model(save_path)
                print(f"best accuracy performence has been updated: {best_score:.5f} --> {dev_score:.5f}")
                best_score = dev_score
            
            # use best score to determine early stopping
            if early_stop:
                if best_score > best_at_epoch_start:
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience:
                    print(
                        f"Early stopping: no validation improvement for {early_stop_patience} epoch(s). "
                        f"best_score={best_score:.5f}"
                    )
                    break
        self.best_score = best_score

    def evaluate(self, data_set, dev_batch_size=None):
        X, y = data_set
        '''
        logits = self.model(X)
        loss = self.loss_fn(logits, y)
        score = self.metric(logits, y)
        return score, loss
        '''
        # If dev_batch_size is None, keep the original single forward on the entire dev set (same as the block above).
        if dev_batch_size is None:
            logits = self.model(X)
            loss = self.loss_fn(logits, y)
            score = self.metric(logits, y)
            return score, loss
        # Weighted averages over chunks: loss_fn returns mean CE per chunk; metric returns mean accuracy per chunk.
        n = X.shape[0]
        bs = max(1, int(dev_batch_size))
        total_loss = 0.0
        total_correct = 0
        for start in range(0, n, bs):
            end = min(start + bs, n)
            logits_b = self.model(X[start:end])
            loss_b = self.loss_fn(logits_b, y[start:end])
            score_b = self.metric(logits_b, y[start:end])
            chunk = end - start
            total_loss += float(loss_b) * chunk
            total_correct += float(score_b) * chunk
        avg_loss = total_loss / n
        avg_score = total_correct / n
        return avg_score, avg_loss
    
    def save_model(self, save_path):
        self.model.save_model(save_path)