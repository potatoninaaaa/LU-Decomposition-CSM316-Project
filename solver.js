document.addEventListener("DOMContentLoaded", () => {

// ================= VALIDATION + PARSING =================

function parseMatrix(text) {
    text = text.trim();
    if (!text) throw new Error("Matrix A cannot be empty. Please enter values.");

    const rows = text.split("\n");
    if (rows.some(r => !r.trim()))
        throw new Error("Matrix A contains empty lines. Remove extra blank lines.");

    const matrix = [];
    rows.forEach((r, idx) => {
        const parts = r.trim().split(/\s+/);
        parts.forEach(p => {
            if (isNaN(Number(p)))
                throw new Error(`Invalid number '${p}' in row ${idx+1}. Only numeric values allowed.`);
        });
        matrix.push(parts.map(Number));
    });

    const n = matrix[0].length;
    matrix.forEach((row, i) => {
        if (row.length !== n)
            throw new Error(`Row ${i+1} has a different number of columns (${row.length}). All rows must be equal.`);
    });

    return matrix;
}

function parseVector(text, expected) {
    text = text.trim();
    if (!text) throw new Error("Vector b cannot be empty. Please enter values.");

    const parts = text.split(/\s+/);
    if (parts.length !== expected)
        throw new Error(`Vector b must have ${expected} elements. You entered ${parts.length}.`);

    const nums = parts.map(Number);
    if (nums.some(isNaN))
        throw new Error("Vector b contains invalid numbers. Only numeric values allowed.");

    return nums;
}

// ================= LU DECOMPOSITION SOLVER WITH PIVOTING =================

function round4(num) {
    return Math.round(num * 10000) / 10000;
}

function luSolver(A, b) {
    const n = A.length;
    const L = Array.from({length: n}, () => Array(n).fill(0));
    const U = Array.from({length: n}, () => Array(n).fill(0));
    const P = Array.from({length: n}, (_, i) => i); // Pivot tracker
    const logs = [];

    logs.push("--- LU Decomposition with Partial Pivoting ---\n");
    logs.push("Legend: U[i,j] = upper triangular, L[i,j] = lower triangular, x[i] = solution, y[i] = intermediate\n");

    for (let i = 0; i < n; i++) {
        // ========== Partial Pivoting ==========
        let maxRow = i;
        let maxVal = Math.abs(A[i][i]);
        for (let k = i + 1; k < n; k++) {
            if (Math.abs(A[k][i]) > maxVal) {
                maxVal = Math.abs(A[k][i]);
                maxRow = k;
            }
        }
        if (maxRow !== i) {
            [A[i], A[maxRow]] = [A[maxRow], A[i]];  // Swap rows in A
            [b[i], b[maxRow]] = [b[maxRow], b[i]];  // Swap rows in b
            [P[i], P[maxRow]] = [P[maxRow], P[i]];  // Track pivot
            logs.push(`Pivoting: swap row ${i+1} with row ${maxRow+1}`);
        }

        // ========== Compute U ==========
        for (let j = i; j < n; j++) {
            let sum = 0;
            let sumStr = "";
            for (let k = 0; k < i; k++) {
                sum += L[i][k] * U[k][j];
                sumStr += ` + (${round4(L[i][k])}·${round4(U[k][j])})`;
            }
            sumStr = sumStr ? sumStr.slice(3) : "0";
            U[i][j] = A[i][j] - sum;
            logs.push(`U[${i+1},${j+1}] = A[${i+1},${j+1}] - Σ(L[${i+1},k]·U[k,${j+1}]) = ${A[i][j]} - (${sumStr}) = ${round4(U[i][j])}`);
        }

        // ========== Compute L ==========
        for (let j = i; j < n; j++) {
            if (i === j) {
                L[i][i] = 1;
                logs.push(`L[${i+1},${i+1}] = 1 (diagonal)`);
            } else {
                let sum = 0;
                let sumStr = "";
                for (let k = 0; k < i; k++) {
                    sum += L[j][k] * U[k][i];
                    sumStr += ` + (${round4(L[j][k])}·${round4(U[k][i])})`;
                }
                sumStr = sumStr ? sumStr.slice(3) : "0";
                L[j][i] = (A[j][i] - sum) / U[i][i];
                logs.push(`L[${j+1},${i+1}] = (A[${j+1},${i+1}] - Σ(L[${j+1},k]·U[k,${i+1}])) / U[${i+1},${i+1}] = (${A[j][i]} - (${sumStr})) / ${round4(U[i][i])} = ${round4(L[j][i])}`);
            }
        }
    }

    // ========== Forward Substitution ==========
    logs.push("\n--- Forward substitution: Ly = b ---");
    const y = Array(n).fill(0);
    for (let i = 0; i < n; i++) {
        let sum = 0;
        let sumStr = "";
        for (let k = 0; k < i; k++) {
            sum += L[i][k] * y[k];
            sumStr += ` + (${round4(L[i][k])}·${round4(y[k])})`;
        }
        sumStr = sumStr ? sumStr.slice(3) : "0";
        y[i] = b[i] - sum;
        logs.push(`y[${i+1}] = b[${i+1}] - Σ(L[${i+1},k]·y[k]) = ${b[i]} - (${sumStr}) = ${round4(y[i])}`);
    }

    // ========== Back Substitution ==========
    logs.push("\n--- Back substitution: Ux = y ---");
    const x = Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
        let sum = 0;
        let sumStr = "";
        for (let k = i + 1; k < n; k++) {
            sum += U[i][k] * x[k];
            sumStr += ` + (${round4(U[i][k])}·${round4(x[k])})`;
        }
        sumStr = sumStr ? sumStr.slice(3) : "0";
        x[i] = (y[i] - sum) / U[i][i];
        logs.push(`x[${i+1}] = (y[${i+1}] - Σ(U[${i+1},k]·x[k])) / U[${i+1},${i+1}] = (${round4(y[i])} - (${sumStr})) / ${round4(U[i][i])} = ${round4(x[i])}`);
    }

    return { 
        L: L.map(r => r.map(round4)), 
        U: U.map(r => r.map(round4)), 
        y: y.map(round4), 
        x: x.map(round4), 
        logs 
    };
}

// ================= BUTTON CLICK HANDLER =================

document.getElementById("runBtn").addEventListener("click", () => {
    const msg = document.getElementById("messages");
    const out = document.getElementById("resultArea");
    msg.textContent = "";
    out.textContent = "";

    try {
        const A = parseMatrix(document.getElementById("matrixA").value);
        const n = A.length;
        const b = parseVector(document.getElementById("vectorB").value, n);

        const { L, U, y, x, logs } = luSolver(A, b);

        out.textContent =
            "LU Decomposition with Partial Pivoting Successful!\n\n" +
            "Step-by-step computation (mathematical expressions):\n" +
            logs.join("\n") + "\n\n" +
            "Lower Triangular Matrix L:\n" + L.map(r => r.join(" ")).join("\n") + "\n\n" +
            "Upper Triangular Matrix U:\n" + U.map(r => r.join(" ")).join("\n") + "\n\n" +
            "Intermediate vector y (Ly = b):\n" + y.join(" ") + "\n\n" +
            "Solution x (Ux = y):\n" + x.join(" ");
    } catch (e) {
        msg.textContent = e.message;
    }
});


//download the python file
const downloadBtn = document.getElementById("downloadPy");

  downloadBtn.addEventListener("click", () => {
    // Fetch the Python file content
    fetch("lu_decomposition.py")
      .then(response => {
        if (!response.ok) throw new Error("File not found");
        return response.blob();
      })
      .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "LU_solver.py"; // Name of the downloaded file
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
      })
      .catch(err => alert("Error downloading file: " + err));
  });
});
