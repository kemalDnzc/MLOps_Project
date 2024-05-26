import pandas as pd
from myapp.models import ExamResult

def load_data_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    for index, row in df.iterrows():
        ExamResult.objects.create(
            gender=row['Gender'],
            ethnic_group=row['EthnicGroup'],
            parent_education=row['ParentEduc'],
            lunch_type=row['LunchType'],
            test_prep=row['TestPrep'],
            math_score=row['MathScore'],
            reading_score=row['ReadingScore'],
            writing_score=row['WritingScore'],
            result_label=row['result_label']
        )

csv_file_path = 'mlops/datasets/cleaned_data.csv'
load_data_from_csv(csv_file_path)
