import numpy as np
import tkinter as tk
from tkinter import scrolledtext

# ================= COLOR PALETTE =================
PRIMARY = "#004E89"
ACCENT = "#F6BC3E"
BLACK = "#000000"
BG = "#F5F7FA"
WHITE = "#FFFFFF"
ERROR_COLOR = "#D32F2F"
SUCCESS_COLOR = "#000000"

# ================= VALIDATION + PARSING =================
def parse_matrix(text):
    text = text.strip()
    if not text:
        raise ValueError("Matrix A cannot be empty. Please enter values.")

    rows = text.split("\n")
    if any(not r.strip() for r in rows):
        raise ValueError("Matrix A contains empty lines. Remove extra blank lines.")

    matrix = []
    for idx, r in enumerate(rows, start=1):
        parts = r.split()
        # detect single number without spaces
        if len(parts) == 1 and len(r.strip()) > 1:
            raise ValueError(f"Row {idx} seems to contain numbers without spaces. Separate columns with spaces.")
        # validate numeric
        for p in parts:
            try:
                float(p)
            except ValueError:
                raise ValueError(f"Invalid number '{p}' in row {idx}. Only numeric values allowed.")
        matrix.append(list(map(float, parts)))

    row_len = len(matrix[0])
    for idx, row in enumerate(matrix, start=1):
        if len(row) != row_len:
            raise ValueError(f"Row {idx} has a different number of columns ({len(row)}). All rows must be equal.")

    return np.array(matrix)

def parse_vector(text, expected_len):
    text = text.strip()
    if not text:
        raise ValueError("Vector b cannot be empty. Please enter values.")
    parts = text.split()
    if len(parts) != expected_len:
        raise ValueError(f"Vector b must have {expected_len} elements. You entered {len(parts)}.")
    try:
        return np.array([float(p) for p in parts])
    except ValueError as e:
        raise ValueError("Vector b contains invalid numbers. Only numeric values allowed.")

# ================= LU DECOMPOSITION =================
def lu_decomposition():
    output.delete("1.0", tk.END)
    try:
        A = parse_matrix(matrix_input.get("1.0", tk.END))
        n = A.shape[0]
        b = parse_vector(vector_input.get("1.0", tk.END), n)

        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix A must be square (n × n).")

        L = np.zeros_like(A, dtype=float)
        U = np.zeros_like(A, dtype=float)

        # LU computation
        for i in range(n):
            # U
            for j in range(i, n):
                U[i][j] = A[i][j] - sum(L[i][k]*U[k][j] for k in range(i))
            if U[i][i] == 0:
                raise ValueError(f"Zero pivot encountered at row {i+1}. Matrix may be singular.")
            L[i][i] = 1
            # L
            for j in range(i+1, n):
                L[j][i] = (A[j][i] - sum(L[j][k]*U[k][i] for k in range(i))) / U[i][i]

        # Forward substitution
        y = np.zeros(n)
        for i in range(n):
            y[i] = b[i] - sum(L[i][k]*y[k] for k in range(i))
        # Backward substitution
        x = np.zeros(n)
        for i in reversed(range(n)):
            x[i] = (y[i] - sum(U[i][k]*x[k] for k in range(i+1, n))) / U[i][i]

        # Display
        output.insert(tk.END, "LU Decomposition Successful!\n\n", "success")
        output.insert(tk.END, f"Matrix A:\n{A}\n\n")
        output.insert(tk.END, f"Vector b:\n{b}\n\n")
        output.insert(tk.END, f"Matrix L:\n{L}\n\n")
        output.insert(tk.END, f"Matrix U:\n{U}\n\n")
        output.insert(tk.END, f"Solution x:\n{x}\n")

    except Exception as e:
        output.insert(tk.END, f"Error: {str(e)}\n", "error")

# ================= CENTER WINDOW =================
def center_window(win):
    win.update_idletasks()
    width = win.winfo_width()
    height = win.winfo_height()
    x = (win.winfo_screenwidth() // 2) - (width // 2)
    y = (win.winfo_screenheight() // 2) - (height // 2)
    win.geometry(f"{width}x{height}+{x}+{y}")

# ================= UI =================
root = tk.Tk()
root.title("LU Decomposition Solver")
root.geometry("780x680")
root.configure(bg=BG)
root.resizable(False, False)
center_window(root)

# Header
header = tk.Frame(root, bg=PRIMARY, pady=18)
header.pack(fill="x")
title = tk.Label(header, text="LU Decomposition Calculator",
                 font=("Segoe UI", 20, "bold"), fg=WHITE, bg=PRIMARY)
title.pack()

# Input Section
container = tk.Frame(root, bg=BG, pady=10)
container.pack()

label_A = tk.Label(container, text="Enter Matrix A (rows separated by newlines, columns by spaces):",
                   font=("Segoe UI", 12, "bold"), bg=BG, fg=BLACK)
label_A.grid(row=0, column=0, sticky="w")
matrix_input = scrolledtext.ScrolledText(container, width=50, height=7, font=("Consolas", 11))
matrix_input.insert("1.0", "Example:\n2 3 1\n4 7 7\n6 18 22")  # placeholder example
matrix_input.grid(row=1, column=0, pady=5)

label_b = tk.Label(container, text="Enter Vector b (space-separated):",
                   font=("Segoe UI", 12, "bold"), bg=BG, fg=BLACK)
label_b.grid(row=2, column=0, sticky="w", pady=(15, 0))
vector_input = scrolledtext.ScrolledText(container, width=50, height=3, font=("Consolas", 11))
vector_input.insert("1.0", "1 2 3")  # placeholder example
vector_input.grid(row=3, column=0, pady=5)

# Button
solve_btn = tk.Button(root, text="Compute LU Decomposition", command=lu_decomposition,
                      bg=ACCENT, fg=BLACK, font=("Segoe UI", 13, "bold"),
                      height=2, width=30, activebackground="#e0a832")
solve_btn.pack(pady=10)

# Output
output_label = tk.Label(root, text="Output:", font=("Segoe UI", 12, "bold"), bg=BG, fg=BLACK)
output_label.pack()
output = scrolledtext.ScrolledText(root, width=95, height=18, font=("Consolas", 11))
output.pack(padx=10, pady=10)

# Tags for color coding
output.tag_config("error", foreground=ERROR_COLOR)
output.tag_config("success", foreground=SUCCESS_COLOR)

root.mainloop()
