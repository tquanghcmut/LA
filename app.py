from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

def givensrotation(a, b):
    hypot = np.sqrt(a**2 + b**2)
    cos = a / hypot
    sin = -b / hypot
    return cos, sin


def qr_givens(A):
    m, n = A.shape
    R = A.copy()
    Q = np.identity(m)
    for i in range(0, n - 1):
        for j in range(i + 1, m):
            cos, sin = givensrotation(R[i, i], R[j, i])
            R[i], R[j] = (R[i] * cos) + (R[j] * (-sin)), (R[i] * sin) + (R[j] * cos)
            Q[:, i], Q[:, j] = (Q[:, i] * cos) + (Q[:, j] * (-sin)), (Q[:, i] * sin) + (Q[:, j] * cos)
    return Q, R


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/decompose', methods=['POST'])
def decompose():
    data = request.form.getlist('matrix[]')
    try:
        matrix = np.array(data, dtype=float).reshape(-1, len(data) // 2)
        Q, R = qr_givens(matrix)
        qqt = np.dot(Q, Q.T)
        qtq = np.dot(Q.T, Q)
        return render_template('result.html', Q=Q, R=R, qqt=qqt, qtq=qtq)
    except ValueError:
        return "Please enter valid numbers in all matrix fields."

if __name__ == '__main__':
    app.run(debug=True)
