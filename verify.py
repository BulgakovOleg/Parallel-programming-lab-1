import sys
import numpy as np

def verify():
    if len(sys.argv) < 5:
        print("Not enough arguments")
        sys.exit(1)
    if len(sys.argv) > 5:
        print("Too many arguments")
        sys.exit(1)

    N = int(sys.argv[1])
    file_A = sys.argv[2]
    file_B = sys.argv[3]
    file_C = sys.argv[4]

    print(f"Loading matrices of size {N}x{N}...")
    
    try:
        A = np.loadtxt(file_A)
        B = np.loadtxt(file_B)
        C_cpp = np.loadtxt(file_C)
    except Exception as e:
        print(f"Failed to read or parse the files. {e}")
        sys.exit(1)

    print("Calculating the reference value using NumPy...")
    C_perfect = np.matmul(A, B)

    print("Comparison of results...")
    if np.allclose(C_cpp, C_perfect, rtol=1e-5, atol=1e-5):
        max_diff = np.max(np.abs(C_cpp - C_perfect))
        print(f"The results of the C++ code match NumPy's exactly!")
        print(f"Maximum deviation of elements: {max_diff:.6f}")
        sys.exit(0)
    else:
        mismatches = np.sum(~np.isclose(C_cpp, C_perfect, rtol=1e-5, atol=1e-5))
        max_diff = np.max(np.abs(C_cpp - C_perfect))
        print(f"Element mismatches {mismatches} detected!")
        print(f"Maximum discrepancy: {max_diff:.6f}")
        sys.exit(1)

if __name__ == "__main__":
    verify()

