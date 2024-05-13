from flask import Flask, render_template, request
import numpy as np
import math

app = Flask(__name__)


def givensrotation(a, b):
    hypot = np.sqrt(a ** 2 + b ** 2)
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


def format_matrix_html(matrix):
    matrix_html = "<div class='matrix'>"
    for row in matrix:
        matrix_html += "<div class='row'>"
        for elem in row:
            matrix_html += f"<div class='cell'>{elem:.2f}</div>"  # Adjust decimal places with :.2f
        matrix_html += "</div>"
    matrix_html += "</div>"
    return matrix_html


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/decompose', methods=['POST'])
def decompose():
    data = request.form.getlist('matrix[]')
    matrix = np.array(data, dtype=float).reshape(int(math.sqrt(len(data))), int(math.sqrt(len(data))))
    Q, R = qr_givens(matrix)
    qqt = np.dot(Q, Q.T)
    qtq = np.dot(Q.T, Q)

    Q_str = np.array2string(Q, formatter={'float_kind': lambda x: "%.3f" % x})
    R_str = np.array2string(R, formatter={'float_kind': lambda x: "%.3f" % x})
    QQT_str = np.array2string(qqt, formatter={'float_kind': lambda x: "%.3f" % x})
    QTQ_str = np.array2string(qtq, formatter={'float_kind': lambda x: "%.3f" % x})


    return render_template('result.html', Q=format_matrix_html(Q), R=format_matrix_html(R),
                       qqt=format_matrix_html(qqt), qtq=format_matrix_html(qtq))

if __name__ == '__main__':
    app.run(debug=True)
