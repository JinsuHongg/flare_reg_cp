import numpy as np

class ConformalPredictor:
    def __init__(self, error: float = 0.05, bins: list = None):
        """
        Initialize the conformal predictor.

        Args:
            error (float): Miscoverage rate (1 - confidence level).
            bins (list): Bin edges for Mondrian CP, e.g., [0, 1, 2, 3, 4, np.inf]
        """
        self.error = error
        self.q_hat = None
        self.q_hat_dict = {}
        self.mode = None
        self.bins = bins

    def q_score(self, cal_arr: np.ndarray, label: np.ndarray, mode: str = 'cqr') -> float | dict:
        """
        Compute the nonconformity score threshold (q_hat or q_hat_dict).

        Args:
            cal_arr (np.ndarray): Calibration predictions.
            label (np.ndarray): True labels.
            mode (str): 'cqr', 'cp', or 'mcp'.

        Returns:
            float or dict: q_hat for CP/CQR or q_hat_dict for MCP.
        """
        if cal_arr.shape[0] != label.shape[0]:
            raise ValueError(f"Mismatch: {cal_arr.shape[0]} calibration vs {label.shape[0]} labels.")

        self.mode = mode.lower()
        n = len(cal_arr)
        q_corrected = np.ceil((1 + n) * (1 - self.error)) / n

        if self.mode == 'cqr':
            if cal_arr.shape[1] != 2:
                raise ValueError(f"CQR expects 2D input, got {cal_arr.shape[1]}D.")
            lower = cal_arr[:, 0] - label[:, 0]
            upper = label[:, 0] - cal_arr[:, 1]
            scores = np.maximum(lower, upper)
            self.q_hat = np.quantile(scores, q=q_corrected, method='higher')
            return self.q_hat

        elif self.mode == 'cp':
            if cal_arr.shape[1] != 1:
                raise ValueError(f"CP expects 1D input, got {cal_arr.shape[1]}D.")
            scores = np.abs(cal_arr[:, 0] - label[:, 0])
            self.q_hat = np.quantile(scores, q=q_corrected, method='higher')
            return self.q_hat

        elif self.mode == "mcp":
            if cal_arr.shape[1] != 1:
                raise ValueError(f"MCP expects 1D input, got {cal_arr.shape[1]}D.")
            if self.bins is None:
                raise ValueError("Bins are required for MCP.")

            scores = np.abs(cal_arr[:, 0] - label[:, 0])
            bin_indices = np.digitize(label.flatten(), bins=self.bins, right=False).reshape(-1, 1)

            for category in np.unique(bin_indices):
                mask = (bin_indices == category).flatten()
                if np.sum(mask) > 0:
                    self.q_hat_dict[category] = np.quantile(scores[mask], q=q_corrected, method='higher')

            return self.q_hat_dict

        else:
            raise ValueError(f"Unknown mode: {mode}. Choose 'cqr', 'cp', or 'mcp'.")

    def pred_regions(self, test_arr: np.ndarray) -> np.ndarray:
        """
        Generate prediction intervals using the computed q_hat or q_hat_dict.

        Args:
            test_arr (np.ndarray): Test predictions.

        Returns:
            np.ndarray: Prediction intervals (lower, upper).
        """
        if self.mode == 'cqr':
            if self.q_hat is None:
                raise ValueError("Call q_score() before pred_regions().")
            lower = test_arr[:, 0] - self.q_hat
            upper = test_arr[:, 1] + self.q_hat
            return np.stack((lower, upper), axis=1)

        elif self.mode == 'cp':
            if self.q_hat is None:
                raise ValueError("Call q_score() before pred_regions().")
            base = test_arr[:, 0]
            lower = base - self.q_hat
            upper = base + self.q_hat
            return np.stack((lower, upper), axis=1)

        elif self.mode == 'mcp':
            if not self.q_hat_dict:
                raise ValueError("Call q_score() before pred_regions().")
            base = test_arr[:, 0]
            regions = np.stack((base, base), axis=1)

            bin_indices = np.digitize(base.flatten(), bins=self.bins, right=False).reshape(-1, 1)

            for category in np.unique(bin_indices):
                mask = (bin_indices == category).flatten()
                q = self.q_hat_dict.get(category, 0)
                regions[mask, 0] -= q
                regions[mask, 1] += q

            return regions

        else:
            raise ValueError(f"Invalid mode: {self.mode}")
