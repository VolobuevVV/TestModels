import ast
import logging
import time
from datetime import datetime
import cv2
import psycopg2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - %(filename)s - %(lineno)d'
)
logger = logging.getLogger()


def get_connection():
    return psycopg2.connect(
        dbname='database',
        user='user',
        password='password',
        host='localhost',
        port='5433'
    )


def get_points_by_time_range(start_time, end_time):
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute('''
                SELECT points FROM plan_points 
                WHERE fixation_time BETWEEN %s AND %s
            ''', (start_time, end_time))
            results = cursor.fetchall()
            points = []
            for row in results:
                points.extend(ast.literal_eval(row[0]))
        return points


def draw_points_on_image(image_path, points):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    for x_rel, y_rel in points:
        x_abs = int(x_rel * width)
        y_abs = int(y_rel * height)
        cv2.circle(image, (x_abs, y_abs), 1, (0, 0, 255), -1)


image_path = "data/plan.webp"
time_start = time.time()
points = get_points_by_time_range(datetime(2025, 9, 16), datetime(2025, 12, 18))
time_end = time.time()
logger.info(f'{time_end - time_start}')
draw_points_on_image(image_path, points)
