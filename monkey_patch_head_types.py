import argparse
import torch
import numpy as np

class HeadType():
    # A registry to map string names to the corresponding class
    _registry = {}

    def __init__(self, name: str):
        self.name = name

    def __init_subclass__(cls, **kwargs):
        """
        This special method is automatically called when a class inherits from HeadType.
        It registers the new subclass in our registry using its exact class name.
        """
        super().__init_subclass__(**kwargs)
        # We only want to register the concrete, usable classes
        if cls.__name__ not in ["HeadType", "ThresholdHeadType"]:
            cls._registry[cls.__name__] = cls

    @classmethod
    def from_string(cls, value: str):
        """
        Parses a string like "DormantHeads:0.5" into a HeadType object.
        This method is designed to be used as a type function in argparse.
        """
        if str(value).lower() == 'none':
            return None

        parts = value.split(':')
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"Invalid format. Expected 'ClassName:value' or 'None', but got '{value}'")

        # The name is expected to be the exact, case-sensitive class name
        name, param_str = parts[0].strip(), parts[1].strip()
        
        if name not in cls._registry:
            raise argparse.ArgumentTypeError(f"Unknown head type '{name}'. Supported types are: {list(cls._registry.keys())}")

        try:
            param = float(param_str)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid parameter value '{param_str}'. Must be a number.")

        # Look up the class in the registry and instantiate it with the parameter
        head_class = cls._registry[name]
        return head_class(param)

class ThresholdHeadType(HeadType):
    def __init__(self, name: str, threshold: float, less_than_threshold: bool):
        super().__init__(name)
        self.set_threshold(threshold)
        self.classname = self.__class__.__name__
        # is the head type defined as less than threshold (true) or greater than threshold (false)?
        self.less_than_threshold = less_than_threshold

    def __str__(self):
        """This is the same representation used when passing this class in as an argument to HeadType"""
        return self.argname
    
    def __repr__(self):
        return self.argname
    
    def generate_all_thresholds(self, model_id: str, task_id: str):
        """Generate a all thresholds for this head type.
            Arg:
                model_id: The model ID to use for generating thresholds. Ex. 'meta-llama/Llama-2-7b-hf'
        """
        model_str = model_id.replace('/', '_')
        cdf_file = f"results-head-score-cdfs/{task_id}/{model_str}_{self.name}.pt"
        cdf = torch.load(cdf_file, weights_only=False)
        ys = np.array([round(p, 2) for p in np.linspace(0.0, 0.3, 7)]) # [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        if not self.less_than_threshold:
            # if our rule is to check if score > threshold, then finding p proportion 
            # of heads corresponds to finding the (1-p) quantile of the CDF
            ys = 1 - ys

        xs = [x_intercept_for_cdf(cdf, y).item() for y in ys]
        return xs
    
    def set_threshold(self, new_threshold: float):
        """Set a new threshold for this head type."""
        assert isinstance(new_threshold, (int, float)) and new_threshold >= 0, "Threshold must be a non-negative number."
        self.threshold = new_threshold
        self.argname = f"{self.__class__.__name__}:{self.threshold}"
        self.labelname = f"{self.__class__.__name__} (Threshold: {self.threshold})"
        

class RandomHeads(ThresholdHeadType):
    """
    Represents a random selection of attention heads. Captures more heads if
    threshold is large (1.0).
    """
    def __init__(self, threshold: float):
        assert 0 <= threshold <= 1, "Threshold (probability) must be between 0 and 1."
        super().__init__(name="Random", threshold=threshold, less_than_threshold=True)
    
    def generate_all_thresholds(self, model_id: str, task_id: str):
        return np.array([round(p, 2) for p in np.linspace(0.0, 0.3, 7)])

class DormantHeads(ThresholdHeadType):
    """
    Represents attention heads where the average attention weight to the
    first token exceeds a threshold. Captures more heads if threshold is 
    small. In the paper, we refer to this as Avg Weight of First Token.

    Ex. threshold=0.8 captures heads where avg attention weight > 0.8
    """
    def __init__(self, threshold: float):
        super().__init__(name="Dormant", threshold=threshold, less_than_threshold=False)

class NormalizedDormantHeads(ThresholdHeadType):
    """
    Represents attention heads where the average attention weight to the
    first token (normalized by avg attention weight to first token across all heads)
    exceeds a threshold. Captures more heads if threshold is small.
    In the paper, we refer to this as Avg Weight of First Token (LN).
    """
    def __init__(self, threshold: float):
        super().__init__(name="NormalizedDormant", threshold=threshold, less_than_threshold=False)

class UnnormalizedHonorHeads(ThresholdHeadType):
    """
    Represents attention heads where the head output norm is less than
    a threshold. Captures more heads if threshold is large.
    In the paper, we refer to this as Avg Head Output Norm.
    """
    def __init__(self, threshold: float):
        super().__init__(name="UnnormalizedHONOR", threshold=threshold, less_than_threshold=True)

class HonorHeads(ThresholdHeadType):
    """
    Represents attention heads where the normalized head output norm is 
    under the layer's average. Captures more heads if threshold is
    large. In the paper, we refer to this as Avg Head Output Norm (LN).

    Ex. threshold=0.1 captures heads with output norms less than
        10% of layer's average.
    """
    def __init__(self, threshold: float):
        super().__init__(name="HONOR", threshold=threshold, less_than_threshold=True)

class FullHeadOutput(ThresholdHeadType):
    """
    Represents attention heads where the head output norm is below
    a threshold. Captures more heads if threshold is large.
    Importantly, a "head output" here refers to multiplying the 
    d_v dimensional (head-dimensional) output vector by a slice of 
    the output projection matrix W_O corresponding to that head.
    """
    def __init__(self, threshold: float):
        super().__init__(name="FullHeadOutput", threshold=threshold, less_than_threshold=True)

class FullHeadOutputNormalized(ThresholdHeadType):
    """
    Represents attention heads where the head output norm is below
    a threshold. Captures more heads if threshold is large.
    Importantly, a "head output" here refers to multiplying the 
    d_v dimensional (head-dimensional) output vector by a slice of 
    the output projection matrix W_O corresponding to that head.
    """
    def __init__(self, threshold: float):
        super().__init__(name="FullHeadOutputNormalized", threshold=threshold, less_than_threshold=True)

class EntropyHeads(ThresholdHeadType):
    """
    Represents attention heads where the average attention entropy is 
    below a threshold. Captures more heads if threshold is large.
    In the paper, we refer to this as Avg Entropy of Query Distributions.

    Ex. threshold=2.0 captures heads with attention entropy less than 2.0
    
    """
    def __init__(self, threshold: float):
        super().__init__(name="Entropy", threshold=threshold, less_than_threshold=True)

class NormalizedEntropyHeads(ThresholdHeadType):
    """
    Represents attention heads where the average attention entropy (normalized
    by average attention entropy across all heads) is below a threshold.
    Captures more heads if threshold is large.
    In the paper, we refer to this as Avg Entropy of Query Distributions (LN).
    """
    def __init__(self, threshold: float):
        super().__init__(name="NormalizedEntropy", threshold=threshold, less_than_threshold=True)

class ValueVectorMagnitudeFirstToken(ThresholdHeadType):
    """
    Represents attention heads where the value vector magnitude
    to the first token is below a threshold. Captures more heads if threshold is large.
    In the paper, we refer to this as First Token Value Vector Norm.
    """
    def __init__(self, threshold: float):
        super().__init__(name="ValueVectorMagnitudeFirstToken", threshold=threshold, less_than_threshold=True)

class ValueVectorMagnitudeNormalizedFirstToken(ThresholdHeadType):
    """
    Represents attention heads where the value vector magnitude
    to the first token (normalized by average value vector magnitude across all heads)
    is below a threshold. Captures more heads if threshold is large.
    In the paper, we refer to this as First Token Value Vector Norm (LN).
    """
    def __init__(self, threshold: float):
        super().__init__(name="ValueVectorMagnitudeNormalizedFirstToken", threshold=threshold, less_than_threshold=True)

class ValueVectorAvgMagnitude(ThresholdHeadType):
    """
    Represents attention heads where the average value vector magnitude
    is below a threshold. Captures more heads if threshold is large.
    In the paper, we refer to this as Avg Value Vector Norm.
    """
    def __init__(self, threshold: float):
        super().__init__(name="ValueVectorAvgMagnitude", threshold=threshold, less_than_threshold=True)

class ValueVectorAvgNormalizedMagnitude(ThresholdHeadType):
    """
    Represents attention heads where the average value vector magnitude
    (normalized by average value vector magnitude across all heads) is below a threshold.
    Captures more heads if threshold is large.
    In the paper, we refer to this as Avg Value Vector Norm (LN).
    """
    def __init__(self, threshold: float):
        super().__init__(name="ValueVectorAvgNormalizedMagnitude", threshold=threshold, less_than_threshold=True)

class HeadOutputMagnitudeLastToken(ThresholdHeadType):
    """
    Represents attention heads where the head output magnitude
    to the last token is below a threshold. Captures more heads if threshold is large.
    In the paper, we refer to this as Last Token Head Output Norm.
    """
    def __init__(self, threshold: float):
        super().__init__(name="HeadOutputMagnitudeLastToken", threshold=threshold, less_than_threshold=True)

class HeadOutputMagnitudeNormalizedLastToken(ThresholdHeadType):
    """
    Represents attention heads where the head output magnitude
    to the last token (normalized by average head output magnitude across all heads)
    is below a threshold. Captures more heads if threshold is large.
    In the paper, we refer to this as Last Token Head Output Norm (LN).
    """
    def __init__(self, threshold: float):
        super().__init__(name="HeadOutputMagnitudeNormalizedLastToken", threshold=threshold, less_than_threshold=True)
    
class HeadOutputMagnitudeNormalizedHeadLastToken(ThresholdHeadType):
    """
    Represents attention heads where the head output magnitude
    to the last token (normalized by average head output of the current head)
    is below a threshold. Captures more heads if threshold is large.
    In the paper, we refer to this as Last Token Head Output Norm (HN).
    """
    def __init__(self, threshold: float):
        super().__init__(name="HeadOutputMagnitudeNormalizedHeadLastToken", threshold=threshold, less_than_threshold=True)

def is_supported_head_type(head_type):
    """Check if the head type is supported."""
    if isinstance(head_type, (RandomHeads, DormantHeads, HonorHeads, NormalizedDormantHeads, UnnormalizedHonorHeads, \
                              EntropyHeads, NormalizedEntropyHeads, ValueVectorMagnitudeFirstToken, ValueVectorMagnitudeNormalizedFirstToken, \
                              ValueVectorAvgMagnitude, ValueVectorAvgNormalizedMagnitude, HeadOutputMagnitudeLastToken, \
                              HeadOutputMagnitudeNormalizedLastToken, HeadOutputMagnitudeNormalizedHeadLastToken,
                              FullHeadOutput, FullHeadOutputNormalized)):
        return True
    return False


# Helper functions

def x_intercept_for_cdf(cdf, y):
    """
    Finds the x-value for a given y-value on a cumulative distribution function (CDF)
    using linear interpolation.

    Args:
        cdf (torch.Tensor): A tensor of shape (N, 2), where the first column contains 
                            the x-values and the second column contains the corresponding 
                            CDF values. The CDF values must be monotonically non-decreasing.
        y (float or torch.Tensor): The scalar or tensor of CDF value(s) to find the x for.
    
    Returns:
        torch.Tensor: The interpolated x-value(s) corresponding to the given y-value(s).
    """
    y = torch.as_tensor(y, dtype=cdf.dtype, device=cdf.device)
    x_vals = cdf[:, 0]
    y_vals = cdf[:, 1]
    
    # Clip y to the range of the provided CDF values to handle edge cases
    # and prevent extrapolation.
    y = torch.clamp(y, min=y_vals[0], max=y_vals[-1])
    
    # Find the indices 'i' of the interval [y_vals[i-1], y_vals[i]] that contains y.
    # torch.searchsorted finds the index of the first element >= y.
    indices = torch.searchsorted(y_vals, y)
    
    # Clamp indices to the valid range [1, N-1] to select the bracketing points.
    indices = torch.clamp(indices, min=1, max=len(y_vals) - 1)
    
    # Get the coordinates of the two points that bracket the target y value.
    x1 = x_vals[indices - 1]
    y1 = y_vals[indices - 1]
    x2 = x_vals[indices]
    y2 = y_vals[indices]
    
    # Perform linear interpolation using the formula: x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    # Calculate the slope of the inverse function (dx/dy).
    denominator = y2 - y1
    
    # To avoid division by zero on flat parts of the CDF (where y1 == y2),
    # replace zeros in the denominator with 1. Since y will equal y1 in these cases,
    # the numerator (y - y1) will be zero, and the fraction correctly evaluates to 0.
    denominator[denominator == 0] = 1.0
    
    # Calculate the interpolation weight.
    fraction = (y - y1) / denominator
    
    # Apply the weight to the x-interval to find the interpolated x-value.
    x_intercept = x1 + fraction * (x2 - x1)
    return x_intercept