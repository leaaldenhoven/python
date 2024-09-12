############################
#     My own questions     #
############################


############################
#          Prep            #

import unicodecsv #library

def read_csv(filename):
    with open(filename,'rb') as f:
        reader = unicodecsv.DictReader(f)
        return list(reader)

enrollments = read_csv('enrollments.csv')
daily_engagement = read_csv('daily_engagement.csv')
project_submissions = read_csv('project_submissions.csv')

from datetime import datetime as dt

def parse_date(date):
    if date == '':
        return None
    else:
        return dt.strptime(date, '%Y-%m-%d')
    
def parse_maybe_int(i):
    if i == '':
        return None
    else:
        return int(i)

for enrollment in enrollments:
    enrollment['cancel_date'] = parse_date(enrollment['cancel_date'])
    enrollment['days_to_cancel'] = parse_maybe_int(enrollment['days_to_cancel'])
    enrollment['is_canceled'] = enrollment['is_canceled'] == 'True'
    enrollment['is_udacity'] = enrollment['is_udacity'] == 'True'
    enrollment['join_date'] = parse_date(enrollment['join_date'])
    
for engagement_record in daily_engagement:
    engagement_record['lessons_completed'] = int(float(engagement_record['lessons_completed']))
    engagement_record['num_courses_visited'] = int(float(engagement_record['num_courses_visited']))
    engagement_record['projects_completed'] = int(float(engagement_record['projects_completed']))
    engagement_record['total_minutes_visited'] = float(engagement_record['total_minutes_visited'])
    engagement_record['utc_date'] = parse_date(engagement_record['utc_date'])
    
for submission in project_submissions:
    submission['completion_date'] = parse_date(submission['completion_date'])
    submission['creation_date'] = parse_date(submission['creation_date'])

for engagement_record in daily_engagement:
    engagement_record["account_key"] = engagement_record["acct"]
    del[engagement_record["acct"]]
daily_engagement[0]['account_key']



## Find the total number of rows and the number of unique students (account keys)

len(enrollments)         # = 1640
len(daily_engagement)    # = 136240
len(project_submissions) # = 3642

def get_unique_students(data):
    unique_students = set()
    for data_point in data:
        unique_students.add(data_point['account_key'])
    return unique_students
        
enrollment_num_unique_students = get_unique_students(enrollments) 
engagement_num_unique_students = get_unique_students(daily_engagement)
submission_num_unique_students = get_unique_students(project_submissions)

## Remove udacity test cases

num_problem_students = 0
for enrollment in enrollments:
    student = enrollment['account_key']
    if student not in engagement_num_unique_students \
            and enrollment['join_date'] != enrollment['cancel_date']:
        num_problem_students += 1

udacity_test_accounts = set()
for enrollment in enrollments:
    if enrollment['is_udacity']:
        udacity_test_accounts.add(enrollment['account_key'])
len(udacity_test_accounts)

def remove_udacity_accounts(data):
    non_udacity_data = []
    for data_point in data:
        if data_point['account_key'] not in udacity_test_accounts:
            non_udacity_data.append(data_point)
    return non_udacity_data
            
non_udacity_enrollments = remove_udacity_accounts(enrollments)
non_udacity_engagement = remove_udacity_accounts(daily_engagement)
non_udacity_submissions = remove_udacity_accounts(project_submissions)        


paid_students = {}
for enrollment in non_udacity_enrollments:
    if not enrollment['is_canceled'] or enrollment['days_to_cancel'] > 7:
        account_key = enrollment['account_key']
        enrollment_date = enrollment['join_date']
        if account_key not in paid_students or \
                enrollment_date > paid_students[account_key]:
            paid_students[account_key] = enrollment_date
len(paid_students)

def within_one_week(join_date, engagement_date):
    time_delta = engagement_date - join_date
    return time_delta.days < 7 and time_delta.days >= 0

def remove_free_trial_cancels(data):
    new_data = []
    for data_point in data:
        if data_point['account_key'] in paid_students:
            new_data.append(data_point)
    return new_data

from collections import defaultdict
def group_data(data, key_name): #what to group it by? we used account key but could use other groupings
    grouped_data = defaultdict(list)
    for data_point in data:
        key = data_point[key_name]
        grouped_data[key].append(data_point)
    return grouped_data

def sum_grouped_items(grouped_data, field_name): #
    summed_data = {}
    for key, data_points in grouped_data.items(): #dictionary of key:data_points
        total = 0
        for data_point in data_points:
            total += data_point[field_name]
        summed_data[key] = total
    return summed_data

import matplotlib.pyplot as plt
import numpy as np  
def describe_data(data):
    print('Mean:', np.mean(data))             
    print('Standard deviation:', np.std(data))
    print('Minimum:', np.min(data))        
    print('Maximum:', np.max(data)) 
    plt.hist(data)    

######## new datasets to be used ########
paid_enrollments = remove_free_trial_cancels(non_udacity_enrollments)
paid_engagement = remove_free_trial_cancels(non_udacity_engagement)
paid_submissions = remove_free_trial_cancels(non_udacity_submissions)
######## new datasets to be used ########

###################################
#       The interesting part      #
###################################

### Q4: For each lesson, how many people pass ###
#       group project_submissions by lesson_key
#       sum up the pass/nonpass in two sets

#HOw many unique courses are there

submission_by_lesson = group_data(paid_submissions, 'lesson_key')

def get_unique_lessons(data):
    unique_lessons = set()
    for data_point in data:
        unique_lessons.add(data_point['lesson_key'])
    return unique_lessons

lessons = get_unique_lessons(paid_submissions) #-> 11 unique lessons
# {'3165188753',
#  '3168208620',
#  '3174288624',
#  '3176718735',
#  '3184238632',
#  '3562208770',
#  '4110338963',
#  '4180859007',
#  '4576183932',
#  '4582204201',
#  '746169184'}

for lesson in lessons:
    lesson_pass = list()
    lesson_nonpass = list()
    lessons_pass_ratios = {}
    for submission in paid_submissions:
        project = submission['lesson_key']
        rating = submission['assigned_rating']
        if project == lesson and (rating == 'PASSED' or rating == 'DISTINCTION'):
            lesson_pass.add(submission['account_key'])
        elif project == lesson and (rating == 'UNGRADED' or rating == 'INCOMPLETE'):
            lesson_nonpass.add(submission['account_key'])
        num_pass = len(lesson_pass)
        num_nonpass = len(lesson_nonpass)
        pass_ratio = num_pass / (num_pass + num_nonpass)
        lessons_pass_ratios[lesson] = pass_ratio

# for one of the lessons python has to 'divide by zero' which I assume means that it might be an empty course?
# passes/nonpasses = 0
# go through section manually for each lesson to find odd-one-out

# {'3165188753',
#  '3168208620',
#  '3174288624',
#  '3176718735',
#  '3184238632',
#  '3562208770',
#  '4110338963',
#  '4180859007',
#  '4576183932',
#  '4582204201',
#  '746169184'}

# lesson_key = '3176718735'
# lesson_pass = set()
# lesson_nonpass = set()
# lessons_pass_ratios = {}
# for submission in paid_submissions:
#     project = submission['lesson_key']
#     rating = submission['assigned_rating']
#     if project == lesson_key and (rating == 'PASSED' or rating == 'DISTINCTION'):
#         lesson_pass.add(submission['account_key'])
#     elif project == lesson_key and (rating == 'UNGRADED' or rating == 'INCOMPLETE'):
#         lesson_nonpass.add(submission['account_key'])
#     else:
#         None
#     num_pass = len(lesson_pass)
#     num_nonpass = len(lesson_nonpass)
#     pass_ratio = num_pass / (num_pass + num_nonpass)
#     lessons_pass_ratios[lesson_key] = pass_ratio


## ??? make a list o all pass_ratios for descriptives?? 
## make a list of two lists? lesson_key and pass_ratios
## dictionary? lesson_key:pass_ratio. how do I add each iteration to it? so I can loop through the lessons

