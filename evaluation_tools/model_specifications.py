"""
Comprehensive model specifications database for YOLO models.
This module contains detailed specifications including GFLOPs, parameters, 
architectural details, and computational complexity for thesis analysis.
"""

import pandas as pd

# Model specifications database
MODEL_SPECS = {
    "YOLOv5u": {
        "parameters_millions": 7.0,
        "gflops_640": 15.8,
        "backbone": "CSPDarknet53",
        "neck": "PANet",
        "head": "YOLOv5 Head",
        "input_channels": 3,
        "architecture_type": "One-stage detector",
        "anchor_based": True,
        "release_year": 2022,
        "paper": "YOLOv5: A State-of-the-Art Real-Time Object Detection System",
        "key_features": "Ultralytics implementation, CSP connections, PANet neck",
        "training_techniques": "Mosaic, MixUp, CutMix, Auto-augmentation"
    },
    "YOLOv8n": {
        "parameters_millions": 3.2,
        "gflops_640": 8.7,
        "backbone": "CSPDarknet53 + C2f",
        "neck": "PANet + C2f",
        "head": "Decoupled Head",
        "input_channels": 3,
        "architecture_type": "Anchor-free detector",
        "anchor_based": False,
        "release_year": 2023,
        "paper": "YOLOv8: A New Real-Time Object Detection System",
        "key_features": "Anchor-free, C2f blocks, decoupled head",
        "training_techniques": "Mosaic, MixUp, CopyPaste, Auto-augmentation"
    },
    "YOLOv10n": {
        "parameters_millions": 2.3,
        "gflops_640": 6.7,
        "backbone": "CSPDarknet53 + C2fCIB",
        "neck": "PANet + C2fCIB",
        "head": "Dual assignments + NMS-free",
        "input_channels": 3,
        "architecture_type": "NMS-free detector",
        "anchor_based": False,
        "release_year": 2024,
        "paper": "YOLOv10: Real-Time End-to-End Object Detection",
        "key_features": "NMS-free, dual label assignments, CIB blocks",
        "training_techniques": "Dual assignments, consistent matching, CIB optimization"
    },
    "YOLO11n": {
        "parameters_millions": 2.6,
        "gflops_640": 6.5,
        "backbone": "CSPDarknet53 + C3k2",
        "neck": "PANet + C3k2",
        "head": "Decoupled Head v2",
        "input_channels": 3,
        "architecture_type": "Anchor-free detector",
        "anchor_based": False,
        "release_year": 2024,
        "paper": "YOLO11: An Improved Real-Time Object Detection System",
        "key_features": "C3k2 blocks, improved FPN, enhanced feature fusion",
        "training_techniques": "Advanced augmentation, improved loss functions"
    }
}

# Export format specifications
EXPORT_FORMAT_SPECS = {
    "PyTorch": {
        "description": "Native PyTorch format",
        "precision": "FP32",
        "optimization_level": "None",
        "hardware_requirements": "CPU/GPU",
        "deployment_complexity": "Easy",
        "inference_backend": "PyTorch"
    },
    "TorchScript": {
        "description": "TorchScript JIT compilation",
        "precision": "FP32",
        "optimization_level": "JIT",
        "hardware_requirements": "CPU/GPU",
        "deployment_complexity": "Easy",
        "inference_backend": "PyTorch C++"
    },
    "ONNX": {
        "description": "Open Neural Network Exchange",
        "precision": "FP32/FP16",
        "optimization_level": "Graph optimization",
        "hardware_requirements": "CPU/GPU/NPU",
        "deployment_complexity": "Medium",
        "inference_backend": "ONNXRuntime"
    },
    "TensorRT": {
        "description": "NVIDIA TensorRT optimization",
        "precision": "FP32/FP16/INT8",
        "optimization_level": "High",
        "hardware_requirements": "NVIDIA GPU",
        "deployment_complexity": "Medium",
        "inference_backend": "TensorRT"
    },
    "MNN": {
        "description": "Mobile Neural Network",
        "precision": "FP32/FP16/INT8",
        "optimization_level": "Mobile optimized",
        "hardware_requirements": "Mobile/ARM",
        "deployment_complexity": "High",
        "inference_backend": "MNN"
    },
    "NCNN": {
        "description": "Tencent NCNN framework",
        "precision": "FP32/FP16/INT8",
        "optimization_level": "Mobile optimized",
        "hardware_requirements": "Mobile/ARM/x86",
        "deployment_complexity": "High",
        "inference_backend": "NCNN"
    }
}

def get_model_specs_df():
    """Returns a DataFrame with detailed model specifications."""
    df = pd.DataFrame.from_dict(MODEL_SPECS, orient='index')
    df.index.name = 'Model'
    return df.reset_index()

def get_export_format_specs_df():
    """Returns a DataFrame with export format specifications."""
    df = pd.DataFrame.from_dict(EXPORT_FORMAT_SPECS, orient='index')
    df.index.name = 'Format'
    return df.reset_index()

def get_model_spec(model_name):
    """Get specifications for a specific model."""
    return MODEL_SPECS.get(model_name, {})

def calculate_theoretical_fps(gflops, hardware_tflops):
    """
    Calculate theoretical maximum FPS based on GFLOPs and hardware capability.
    
    Args:
        gflops: Model complexity in GFLOPs
        hardware_tflops: Hardware capability in TFLOPs
    
    Returns:
        Theoretical maximum FPS
    """
    return (hardware_tflops * 1000) / gflops

def get_efficiency_metrics(model_name, actual_fps, actual_map):
    """
    Calculate efficiency metrics for model comparison.
    
    Args:
        model_name: Name of the model
        actual_fps: Measured FPS
        actual_map: Measured mAP
    
    Returns:
        Dictionary with efficiency metrics
    """
    specs = get_model_spec(model_name)
    if not specs:
        return {}
    
    params = specs.get('parameters_millions', 1)
    gflops = specs.get('gflops_640', 1)
    
    return {
        'fps_per_gflop': actual_fps / gflops,
        'fps_per_param': actual_fps / params,
        'map_per_gflop': actual_map / gflops,
        'map_per_param': actual_map / params,
        'efficiency_score': (actual_map * actual_fps) / (gflops * params)
    }

# Hardware specifications for reference
HARDWARE_SPECS = {
    "RTX_3060": {
        "fp32_tflops": 13.0,
        "fp16_tflops": 26.0,
        "tensor_tflops": 51.0,
        "memory_gb": 12,
        "memory_bandwidth_gbps": 360,
        "cuda_cores": 3584
    },
    "RTX_4090": {
        "fp32_tflops": 33.0,
        "fp16_tflops": 66.0,
        "tensor_tflops": 165.0,
        "memory_gb": 24,
        "memory_bandwidth_gbps": 1008,
        "cuda_cores": 16384
    },
    "Jetson_Orin": {
        "fp32_tflops": 1.7,
        "fp16_tflops": 3.4,
        "tensor_tflops": 27.0,
        "memory_gb": 8,
        "memory_bandwidth_gbps": 102,
        "cuda_cores": 1024
    },
    "CPU_i7_12700": {
        "fp32_tflops": 0.4,
        "fp16_tflops": 0.8,
        "tensor_tflops": 0.0,
        "memory_gb": 32,
        "memory_bandwidth_gbps": 51.2,
        "cores": 12
    }
}

def get_hardware_specs_df():
    """Returns a DataFrame with hardware specifications."""
    df = pd.DataFrame.from_dict(HARDWARE_SPECS, orient='index')
    df.index.name = 'Hardware'
    return df.reset_index()

if __name__ == "__main__":
    # Example usage
    print("Model Specifications:")
    print(get_model_specs_df())
    print("\nExport Format Specifications:")
    print(get_export_format_specs_df())
    print("\nHardware Specifications:")
    print(get_hardware_specs_df()) 