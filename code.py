# alpha_framework.py

# === DATA ABSTRACTION ===

class DataSource:
    """
    Abstract base class for any data source (OHLCV, factors, alt-data, etc.).
    Each model may use a different type of data.
    """
    def __init__(self, config):
        self.config = config

    def get_symbols(self):
        raise NotImplementedError

    def get_universe(self, date):
        return None

    def get_slice(self, start_date, end_date):
        raise NotImplementedError


class DataLoader:
    """
    Manages multiple DataSources and handles slicing/caching of data.
    """
    def __init__(self, sources):
        self.sources = sources  # Dict of name → DataSource

    def get(self, name, start_date, end_date):
        return self.sources[name].get_slice(start_date, end_date)


# === FEATURE PIPELINE ===

class FeaturePipeline:
    """
    Stateless feature pipeline: applies a series of transformers to data.
    No fit/fit_transform needed.
    """
    def __init__(self, transformers):
        self.transformers = transformers  # List of callables

    def apply(self, data):
        for t in self.transformers:
            data = t(data)
        return data


# Example transformer
class Normalize:
    """
    Example transformer that normalizes the data.
    """
    def __init__(self, method="zscore"):
        self.method = method

    def __call__(self, data):
        # Apply normalization here (placeholder)
        return data


# === MODEL INTERFACE ===

class BaseModel:
    """
    Abstract base class for all models.
    Each model must implement forward and loss functions.
    """
    def __init__(self, config):
        self.config = config

    def forward(self, batch):
        raise NotImplementedError

    def loss(self, predictions, targets):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError


# === TRAINING ENGINE ===

class Trainer:
    """
    Handles custom training loop.
    Supports batching, streaming, checkpointing, etc.
    """
    def __init__(self, model, data_loader, optimizer, config):
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.config = config
        self.report = None

    def train(self):
        """
        Training loop with custom batch logic.
        Can store training metrics to ReportManager.
        """
        for epoch in range(self.config["epochs"]):
            for batch in self.data_loader:
                preds = self.model.forward(batch["X"])
                loss = self.model.loss(preds, batch["y"])
                self.optimizer.step(loss)

                # Optionally record loss per batch
                if self.report:
                    self.report.log_metric("train_loss", loss, step=epoch)


# === BACKTESTING ENGINE ===

class BacktestEngine:
    """
    Evaluates model on in-sample or out-of-sample data.
    """
    def __init__(self, model, dataset, config):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.report = None

    def run(self):
        """
        Apply model to test dataset and store predictions.
        """
        # Store predictions and targets for later evaluation
        predictions = []
        targets = []

        for batch in self.dataset:
            pred = self.model.forward(batch["X"])
            predictions.append(pred)
            targets.append(batch["y"])

        if self.report:
            self.report.log_array("predictions", predictions)
            self.report.log_array("targets", targets)

    def evaluate(self):
        """
        Compute evaluation metrics (placeholder).
        """
        metrics = {
            "sharpe": None,
            "ir": None,
            "drawdown": None
        }

        if self.report:
            self.report.log_metrics(metrics)


# === REPORTING ===

class ReportManager:
    """
    Central structure to save training, testing, and alpha data.
    Can export to disk or UI.
    """
    def __init__(self, model_name):
        self.model_name = model_name
        self.data = {
            "metrics": {},
            "arrays": {},
            "info": {}
        }

    def log_metric(self, name, value, step=None):
        if name not in self.data["metrics"]:
            self.data["metrics"][name] = []
        self.data["metrics"][name].append((step, value))

    def log_metrics(self, metrics: dict):
        for k, v in metrics.items():
            self.log_metric(k, v)

    def log_array(self, name, values):
        self.data["arrays"][name] = values

    def add_info(self, key, value):
        self.data["info"][key] = value

    def save(self, path):
        """
        Serialize to JSON, CSV, etc.
        """
        pass


# === MODEL COMPARISON ===

class ModelComparer:
    """
    Compare models on performance metrics.
    """
    def __init__(self, reports):
        self.reports = reports  # List of ReportManager

    def rank(self, metric_name, higher_is_better=True):
        ranking = []
        for report in self.reports:
            values = report.data["metrics"].get(metric_name)
            if not values:
                continue
            final_value = values[-1][1]  # Last logged value
            ranking.append((report.model_name, final_value))

        ranking.sort(key=lambda x: x[1], reverse=higher_is_better)
        return ranking


# === EXPERIMENT RUNNER ===

class ExperimentRunner:
    """
    Top-level orchestrator for end-to-end runs.
    """
    def __init__(self, config):
        self.config = config

    def run(self):
        # 1. Load and slice data using DataLoader
        # 2. Apply feature pipeline
        # 3. Initialize model and trainer
        # 4. Run training
        # 5. Run backtest and evaluation
        # 6. Save report
        pass
