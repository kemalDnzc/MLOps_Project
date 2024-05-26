from django.db import models

class ExamResult(models.Model):
    gender = models.CharField(max_length=10)
    ethnic_group = models.CharField(max_length=20)
    parent_education = models.CharField(max_length=50)
    lunch_type = models.CharField(max_length=20)
    test_prep = models.CharField(max_length=20)
    math_score = models.IntegerField()
    reading_score = models.IntegerField()
    writing_score = models.IntegerField()

    def __str__(self):
        return f'{self.gender} - {self.ethnic_group} - {self.parent_education}'
