import numpy as np
import csv


def csv_to_numpy_array(file_path):
    """
    Reads a CSV file and converts its data into a NumPy array, replacing non-numeric values with 0.
    The first row (header) is excluded.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        numpy.ndarray: A NumPy array with numeric data (non-numeric values replaced by 0).
    """
    numeric_data = []

    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)

        # 첫 줄(헤더) 건너뛰기
        next(reader, None)

        for row in reader:
            numeric_row = []
            for item in row:
                try:
                    # 숫자로 변환
                    numeric_row.append(float(item))
                except ValueError:
                    # 변환 실패 시 0 추가
                    numeric_row.append(0.0)
            numeric_data.append(numeric_row)

    return np.array(numeric_data)


# 사용 예시
file_path = 'Geometry.csv'  # CSV 파일 경로
result_array = csv_to_numpy_array(file_path)
print(result_array)