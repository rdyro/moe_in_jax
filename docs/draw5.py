import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_final_corrected_pipeline():
    # --- Configuration ---
    num_batches = 4

    # Time units
    # Requirement: Send + Recv < Compute (to show slack)
    # Requirement: Send and Recv do not overlap (sequential contention)
    w_send = 1.5
    w_recv = 1.5
    w_compute = 4.0
    # Total comms = 3.0, Compute = 4.0. Slack = 1.0.

    # Layout dimensions
    h_bar = 0.5
    y_spacing = 1.0

    # Vertical Positions
    y_title = 6.2
    y_logical_row = 5.5
    y_pipe_header = 4.2
    y_pipe_start = 3.5

    # Colors
    c_send = "#DAE8FC"  # Blue
    c_compute = "#FFF2CC"  # Yellow
    c_recv = "#D5E8D4"  # Green
    c_stroke = "#333333"  # Dark Grey

    fig, ax = plt.subplots(figsize=(16, 8), dpi=100)

    # ==========================================
    # 1. Logical Task Flow (Top)
    # ==========================================
    ax.text(
        -0.5, y_title, "Logical Task Flow", fontsize=12, fontweight="bold", color="#333"
    )

    patches_logical = [
        (0, "ra2a\nsend", c_send, w_send),
        (w_send, "compute", c_compute, w_compute),
        (w_send + w_compute, "ra2a\nreceive", c_recv, w_recv),
    ]

    for x, lbl, col, w in patches_logical:
        rect = patches.Rectangle(
            (x, y_logical_row),
            w,
            h_bar,
            linewidth=1.5,
            edgecolor=c_stroke,
            facecolor=col,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y_logical_row + h_bar / 2,
            lbl,
            ha="center",
            va="center",
            fontsize=9,
        )

    # Divider Line
    ax.plot(
        [-1, 22],
        [y_pipe_header + 0.5, y_pipe_header + 0.5],
        color="#888",
        linestyle="--",
        linewidth=1,
    )

    # ==========================================
    # 2. Pipelined Execution (Bottom)
    # ==========================================
    ax.text(
        -0.5,
        y_pipe_header,
        f"Pipelined Execution ({num_batches} Micro-batches)",
        fontsize=12,
        fontweight="bold",
        color="#333",
    )

    # --- Timing Calculations ---
    # We anchor everything to the Compute blocks because they are the bottleneck.
    # Compute blocks run back-to-back starting after the first Send.

    comp_starts = []
    comp_ends = []

    # Batch 0 Compute starts exactly when Batch 0 Send ends
    current_c_start = w_send

    for i in range(num_batches):
        comp_starts.append(current_c_start)
        comp_ends.append(current_c_start + w_compute)
        current_c_start += w_compute  # Serialized execution

    # Draw Batches
    for i in range(num_batches):
        y = y_pipe_start - (i * y_spacing)

        # --- A. Receive Block ---
        # Starts immediately after THIS batch's compute finishes
        r_start = comp_ends[i]
        rect_r = patches.Rectangle(
            (r_start, y),
            w_recv,
            h_bar,
            linewidth=1.5,
            edgecolor=c_stroke,
            facecolor=c_recv,
        )
        ax.add_patch(rect_r)
        ax.text(
            r_start + w_recv / 2,
            y + h_bar / 2,
            f"recv {i + 1}",
            ha="center",
            va="center",
            fontsize=8,
        )

        # --- B. Compute Block ---
        c_start = comp_starts[i]
        rect_c = patches.Rectangle(
            (c_start, y),
            w_compute,
            h_bar,
            linewidth=1.5,
            edgecolor=c_stroke,
            facecolor=c_compute,
        )
        ax.add_patch(rect_c)
        ax.text(
            c_start + w_compute / 2,
            y + h_bar / 2,
            f"compute {i + 1}",
            ha="center",
            va="center",
            fontsize=8,
        )

        # --- C. Send Block ---
        # Determine Start Time based on contention logic
        if i == 0:
            s_start = 0
        elif i == 1:
            # Batch 2 Send happens inside Batch 1 Compute.
            # There is no "Recv Batch 0" to contend with.
            # It starts as soon as Batch 1 Compute starts.
            s_start = comp_starts[i - 1]
        else:
            # Batch N Send (where N > 2) happens inside Batch N-1 Compute.
            # However, Batch N-2 Receive is ALSO happening inside Batch N-1 Compute.
            # Contention Rule: Recv(N-2) first, then Send(N).

            # Recv(i-2) starts at comp_ends[i-2].
            # Note: comp_ends[i-2] is exactly comp_starts[i-1].
            # So Recv(i-2) occupies [comp_starts[i-1], comp_starts[i-1] + w_recv].

            # Therefore, Send(i) must wait until Recv(i-2) finishes.
            s_start = comp_starts[i - 1] + w_recv

        rect_s = patches.Rectangle(
            (s_start, y),
            w_send,
            h_bar,
            linewidth=1.5,
            edgecolor=c_stroke,
            facecolor=c_send,
        )
        ax.add_patch(rect_s)
        ax.text(
            s_start + w_send / 2,
            y + h_bar / 2,
            f"send {i + 1}",
            ha="center",
            va="center",
            fontsize=8,
        )

    # ==========================================
    # 3. Vertical Markers & Labels
    # ==========================================

    # Helper for dashed lines
    def draw_line(x):
        plt.vlines(
            x=x,
            ymin=-0.5,
            ymax=y_pipe_start + h_bar,
            colors="#444",
            linestyles="--",
            linewidth=1.5,
        )

    # Line 1: End of Preamble (Start of first Compute)
    preamble_end_x = comp_starts[0]
    draw_line(preamble_end_x)

    # Label: "Preamble" -> Centered in the first column (0 to 1.5)
    ax.text(
        preamble_end_x / 2,
        -0.5,
        "preamble",
        ha="center",
        va="top",
        fontsize=12,
        fontstyle="italic",
        color="#444",
    )

    # Line 2, 3...: Steady states (End of computes)
    for idx, x in enumerate(comp_ends):
        draw_line(x)

        # Label: "Epilogue" -> Under the last section
        if idx == num_batches - 1:
            # Center the label between end of last compute and end of last receive
            epilogue_center = x + (w_recv / 2)
            ax.text(
                epilogue_center,
                -0.5,
                "epilogue",
                ha="center",
                va="top",
                fontsize=12,
                fontstyle="italic",
                color="#444",
            )

    # --- Final Layout ---
    total_width = comp_ends[-1] + w_recv
    ax.set_xlim(-1, total_width + 1)
    ax.set_ylim(-1.5, y_title + 1)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("out.svg")
    plt.show()


if __name__ == "__main__":
    draw_final_corrected_pipeline()
