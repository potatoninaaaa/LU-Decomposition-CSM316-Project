import numpy as np
import tkinter as tk
from tkinter import scrolledtext

# ================= COLOR PALETTE =================
PRIMARY_BG = "#e6e6e3"
CARD_BG = "#d8dcd9"
INPUT_BG = "#f0f2f0"
TEXT_DARK = "#2f4f54"
ACCENT_BTN = "#355a60"
OUTPUT_BG = "#cfd5d1"
ERROR_COLOR = "#a7422c"

# ================= VALIDATION + PARSING =================
def parse_matrix(text):
    text = text.strip()
    if not text:
        raise ValueError("Matrix A cannot be empty.")

    rows = text.split("\n")
    if any(not r.strip() for r in rows):
        raise ValueError("Matrix A contains empty lines.")

    matrix = []
    for idx, r in enumerate(rows, start=1):
        parts = r.split()

        # Detect concatenated numbers (no spaces)
        if len(parts) == 1 and len(r.strip()) > 1:
            raise ValueError(f"Row {idx} looks incorrect. Use spaces between numbers.")

        for p in parts:
            try:
                float(p)
            except ValueError:
                raise ValueError(f"Invalid number '{p}' in row {idx}.")
        matrix.append(list(map(float, parts)))

    col_count = len(matrix[0])
    for idx, row in enumerate(matrix, start=1):
        if len(row) != col_count:
            raise ValueError(f"Row {idx} has inconsistent column count.")

    return np.array(matrix)


def parse_vector(text, expected_len):
    text = text.strip()
    if not text:
        raise ValueError("Vector b cannot be empty.")

    parts = text.split()
    if len(parts) != expected_len:
        raise ValueError(f"Vector b must have {expected_len} elements.")

    try:
        return np.array([float(p) for p in parts])
    except ValueError:
        raise ValueError("Vector b contains invalid numbers.")


# ================= LU DECOMPOSITION =================
def lu_decomposition():
    output.delete("1.0", tk.END)
    try:
        A = parse_matrix(matrix_input.get("1.0", tk.END))
        n = A.shape[0]
        b = parse_vector(vector_input.get("1.0", tk.END), n)

        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix A must be square.")

        L = np.zeros_like(A, float)
        U = np.zeros_like(A, float)
        P = np.arange(n)
        logs = []

        logs.append("--- LU Decomposition with Partial Pivoting ---\n")

        for i in range(n):
            # Pivoting
            max_row = i + np.argmax(np.abs(A[i:, i]))
            if max_row != i:
                A[[i, max_row]] = A[[max_row, i]]
                b[i], b[max_row] = b[max_row], b[i]
                P[i], P[max_row] = P[max_row], P[i]
                logs.append(f"Swapped rows {i+1} and {max_row+1}")

            # Fill U row i
            for j in range(i, n):
                U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))

            # Fill L column i
            L[i, i] = 1
            for j in range(i + 1, n):
                L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]

        # Forward Substitution
        y = np.zeros(n)
        for i in range(n):
            y[i] = b[i] - sum(L[i, k] * y[k] for k in range(i))

        # Backward Substitution
        x = np.zeros(n)
        for i in reversed(range(n)):
            x[i] = (y[i] - sum(U[i, k] * x[k] for k in range(i + 1, n))) / U[i, i]

        # Output
        output.insert(tk.END, "LU Decomposition Successful!\n\n")
        output.insert(tk.END, f"L matrix:\n{np.round(L, 4)}\n\n")
        output.insert(tk.END, f"U matrix:\n{np.round(U, 4)}\n\n")
        output.insert(tk.END, f"y vector:\n{np.round(y, 4)}\n\n")
        output.insert(tk.END, f"x vector:\n{np.round(x, 4)}\n\n")
        output.insert(tk.END, "Steps:\n" + "\n".join(logs))

    except Exception as e:
        output.insert(tk.END, f"Error: {str(e)}\n", "error")


# ================= UI =================
root = tk.Tk()
root.title("LU Decomposition Solver")
root.geometry("1300x720")
root.configure(bg=PRIMARY_BG)
root.resizable(False, False)

# Main wrapper
main_frame = tk.Frame(root, bg=PRIMARY_BG, padx=40, pady=40)
main_frame.pack(fill="both", expand=True)

# Title
title = tk.Label(
    main_frame,
    text="LU Decomposition Solver",
    font=("Arial", 42, "bold"),
    fg=TEXT_DARK,
    bg=PRIMARY_BG
)
title.pack(pady=(0, 10))

subtitle = tk.Label(
    main_frame,
    text="Enter matrix A and vector b to compute the LU decomposition with partial pivoting.",
    font=("Arial", 16),
    fg="#445c60",
    bg=PRIMARY_BG
)
subtitle.pack(pady=(0, 30))

# Two columns
columns = tk.Frame(main_frame, bg=PRIMARY_BG)
columns.pack()

# LEFT COLUMN (input)
left = tk.Frame(columns, bg=PRIMARY_BG)
left.pack(side="left", padx=25)

card = tk.Frame(left, bg=CARD_BG, padx=36, pady=32, relief=tk.FLAT)
card.pack()

# Matrix A
label_A = tk.Label(card, text="Matrix A:", font=("Arial", 16, "bold"), bg=CARD_BG, fg=TEXT_DARK)
label_A.pack(anchor="w")

matrix_input = scrolledtext.ScrolledText(
    card, width=45, height=7, font=("Consolas", 12),
    bg=INPUT_BG, fg=TEXT_DARK, border=2, relief="solid"
)
matrix_input.insert("1.0", "4 12 -16\n12 37 -43\n-16 -43 98")
matrix_input.pack(pady=5)

# Vector b
label_b = tk.Label(card, text="Vector b:", font=("Arial", 16, "bold"), bg=CARD_BG, fg=TEXT_DARK)
label_b.pack(anchor="w", pady=(15, 5))

vector_input = scrolledtext.ScrolledText(
    card, width=45, height=3, font=("Consolas", 12),
    bg=INPUT_BG, fg=TEXT_DARK, border=2, relief="solid"
)
vector_input.insert("1.0", "1 2 3")
vector_input.pack(pady=5)

# Solve Button
solve_btn = tk.Button(
    card,
    text="Compute LU Decomposition",
    command=lu_decomposition,
    bg=ACCENT_BTN, fg="white",
    font=("Arial", 15, "bold"),
    padx=20, pady=10,
    relief="flat"
)
solve_btn.pack(pady=10)

# RIGHT COLUMN (output)
right = tk.Frame(columns, bg=PRIMARY_BG)
right.pack(side="right", padx=25)

output_label = tk.Label(
    right, text="Output:", font=("Arial", 18, "bold"),
    fg=TEXT_DARK, bg=PRIMARY_BG
)
output_label.pack(anchor="w")

output = scrolledtext.ScrolledText(
    right, width=60, height=27, font=("Consolas", 11),
    bg=OUTPUT_BG, fg="#2b3b3d",
    border=2, relief="solid"
)
output.pack(pady=10)

root.mainloop()
