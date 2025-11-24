"""
Visualization script for model predictions.
This script can visualize predictions saved by inference.py in save_samples mode.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os


def visualize_predictions(
    inputs_path,
    labels_path,
    outputs_path,
    output_dir=None,
    num_samples=4,
    channel_names=None,
    cmap="gist_ncar",
):
    """
    Visualize model predictions vs ground truth labels.

    Args:
        inputs_path: Path to inputs.npy file
        labels_path: Path to labels.npy file
        outputs_path: Path to outputs.npy file
        output_dir: Directory to save visualization images. If None, saves in the same directory as inputs.
        num_samples: Number of samples to visualize (default: 4)
        channel_names: List of channel names for labeling. If None, uses default names.
        cmap: Colormap to use (default: "gist_ncar")
    """
    # Load data
    inputs = np.load(inputs_path)
    labels = np.load(labels_path)
    outputs = np.load(outputs_path)

    print(f"Loaded data shapes:")
    print(f"  Inputs: {inputs.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Outputs: {outputs.shape}")

    # Determine number of samples to visualize
    num_available = min(inputs.shape[0], labels.shape[0], outputs.shape[0])
    num_samples = min(num_samples, num_available)

    # Select random samples
    if num_samples < num_available:
        import random
        indices = random.sample(range(num_available), num_samples)
    else:
        indices = list(range(num_samples))

    inputs_vis = inputs[indices]
    labels_vis = labels[indices]
    outputs_vis = outputs[indices]

    # Determine number of channels
    if len(inputs.shape) == 4:  # (batch, channels, height, width)
        num_channels = inputs.shape[1]
    elif len(inputs.shape) == 3:  # (batch, height, width) - single channel
        num_channels = 1
        inputs_vis = inputs_vis[:, np.newaxis, :, :]
        labels_vis = labels_vis[:, np.newaxis, :, :]
        outputs_vis = outputs_vis[:, np.newaxis, :, :]
    else:
        raise ValueError(f"Unexpected input shape: {inputs.shape}")

    # Default channel names
    if channel_names is None:
        if num_channels == 1:
            channel_names = ["Channel 0"]
        elif num_channels == 4:
            channel_names = ["rho", "u", "v", "p"]
        elif num_channels == 5:
            channel_names = ["rho", "u", "v", "p", "tracer"]
        else:
            channel_names = [f"Channel {i}" for i in range(num_channels)]

    # Create visualization for each channel
    for channel_idx in range(num_channels):
        fig = plt.figure(figsize=(4 * num_samples, 6))
        grid = ImageGrid(
            fig,
            111,
            nrows_ncols=(3, num_samples),
            axes_pad=0.1,
            cbar_location="right",
            cbar_mode="single",
            cbar_size="5%",
            cbar_pad=0.1,
        )

        # Get data for this channel
        inputs_channel = inputs_vis[:, channel_idx, :, :]
        labels_channel = labels_vis[:, channel_idx, :, :]
        outputs_channel = outputs_vis[:, channel_idx, :, :]

        # Compute global min/max for consistent color scale
        vmax = max(outputs_channel.max(), labels_channel.max(), inputs_channel.max())
        vmin = min(outputs_channel.min(), labels_channel.min(), inputs_channel.min())

        # Plot inputs, labels, and outputs
        for sample_idx in range(num_samples):
            # Input
            ax = grid[sample_idx]
            im = ax.imshow(
                inputs_channel[sample_idx],
                cmap=cmap,
                origin="lower",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if sample_idx == 0:
                ax.set_ylabel("Input", fontsize=12)

            # Label (ground truth)
            ax = grid[num_samples + sample_idx]
            ax.imshow(
                labels_channel[sample_idx],
                cmap=cmap,
                origin="lower",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if sample_idx == 0:
                ax.set_ylabel("Ground Truth", fontsize=12)

            # Output (prediction)
            ax = grid[2 * num_samples + sample_idx]
            ax.imshow(
                outputs_channel[sample_idx],
                cmap=cmap,
                origin="lower",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if sample_idx == 0:
                ax.set_ylabel("Prediction", fontsize=12)

        # Add colorbar
        cbar = grid.cbar_axes[0].colorbar(im)
        cbar.set_label("Value", rotation=270, labelpad=15)

        # Set title
        fig.suptitle(f"Channel: {channel_names[channel_idx]}", fontsize=14, y=0.98)

        # Save figure
        if output_dir is None:
            output_dir = os.path.dirname(inputs_path)
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f"visualization_channel_{channel_idx}.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to: {output_path}")
        plt.close()

    # Create error visualization (prediction - ground truth)
    for channel_idx in range(num_channels):
        fig = plt.figure(figsize=(4 * num_samples, 4))
        grid = ImageGrid(
            fig,
            111,
            nrows_ncols=(1, num_samples),
            axes_pad=0.1,
            cbar_location="right",
            cbar_mode="single",
            cbar_size="5%",
            cbar_pad=0.1,
        )

        labels_channel = labels_vis[:, channel_idx, :, :]
        outputs_channel = outputs_vis[:, channel_idx, :, :]
        errors = outputs_channel - labels_channel

        # Use symmetric color scale for errors
        error_max = np.abs(errors).max()
        vmin_error = -error_max
        vmax_error = error_max

        for sample_idx in range(num_samples):
            ax = grid[sample_idx]
            im = ax.imshow(
                errors[sample_idx],
                cmap="RdBu_r",
                origin="lower",
                vmin=vmin_error,
                vmax=vmax_error,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if sample_idx == 0:
                ax.set_ylabel("Error\n(Pred - GT)", fontsize=12)

        cbar = grid.cbar_axes[0].colorbar(im)
        cbar.set_label("Error", rotation=270, labelpad=15)

        fig.suptitle(f"Prediction Error: {channel_names[channel_idx]}", fontsize=14, y=0.98)

        output_path = os.path.join(output_dir, f"error_channel_{channel_idx}.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved error visualization to: {output_path}")
        plt.close()

    print(f"\nVisualization complete! All images saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize model predictions from saved .npy files"
    )
    parser.add_argument(
        "--inputs",
        type=str,
        required=True,
        help="Path to inputs.npy file",
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Path to labels.npy file",
    )
    parser.add_argument(
        "--outputs",
        type=str,
        required=True,
        help="Path to outputs.npy file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save visualization images. If not specified, saves in the same directory as inputs.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="Number of samples to visualize (default: 4)",
    )
    parser.add_argument(
        "--channel_names",
        type=str,
        nargs="+",
        default=None,
        help="Names of channels (e.g., rho u v p). If not specified, uses defaults based on number of channels.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="gist_ncar",
        help="Colormap to use (default: gist_ncar)",
    )

    args = parser.parse_args()

    visualize_predictions(
        inputs_path=args.inputs,
        labels_path=args.labels,
        outputs_path=args.outputs,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        channel_names=args.channel_names,
        cmap=args.cmap,
    )

