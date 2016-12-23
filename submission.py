#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 14:27:12 2016
@author: antoine
"""

import pandas as pd
from collections import OrderedDict

class Assignment:
    def __init__(self,
                 assignment_name,
                 submission_file='./submission.txt',
                 train_dir='./data/CSPL_RECEIVED_CALLS_series/',
                 featureExtractor=None,
                 regressor=None,
                 verbose=True):
        self.assignment_name = assignment_name
        self.submission_file = submission_file
        self.train_dir = train_dir
        self.verbose = verbose
        
        #####
        self.assignment_name_column = None
        self.submission = None
        self.train = None
        self.assignment = None
        
        #####
        self.featureExtractor = featureExtractor
        self.regressor = regressor
        
        self.X_array = None
        
        self.preds_full_range = None
        self.preds_submission = None
        
    def set_submission(self):
        """
        Load full submission file
        Suboptimal but it is a small file
        """
        raw_submission = pd.read_csv(self.submission_file,
                                      sep='\t',
                                      parse_dates=[0]).groupby('ASS_ASSIGNMENT')
        
        # Get right assignment
        self.assignment_name_column = raw_submission.get_group(self.assignment_name)['ASS_ASSIGNMENT'].reset_index()
        assignment = raw_submission.get_group(self.assignment_name).drop('ASS_ASSIGNMENT', 1)
        assignment = assignment.rename(columns={"prediction": "CSPL_RECEIVED_CALLS"})
        assignment['CSPL_RECEIVED_CALLS'] = range(1, len(assignment['CSPL_RECEIVED_CALLS']) + 1)
        assignment = assignment.set_index('DATE')
        self.submission = assignment
        return True
    
    def set_train(self):
        """
        Load training file
        """
        assignment = pd.read_csv(self.train_dir + self.assignment_name + '.csv',
                                 sep=';',
                                 parse_dates=[0])
        assignment = assignment.set_index('DATE')
        self.train = assignment
        return True
        
    def set_assignment(self):
        assignment = pd.concat([self.submission, self.train], axis=0)
        assignment = assignment.sort_index().reset_index()
        self.assignment = assignment
        return True
    
    def preprocess(self):
        self.set_submission()
        self.set_train()
        self.set_assignment()
        if self.verbose: print '[OK] Preprocessing - %s'%self.assignment_name
    
    def transform(self):
        # self.X_array = self.featureExtractor.transform(self.assignment)
        if self.verbose: print '[OK] Feature engineering - %s'%self.assignment_name
        return True
    
    def predict(self):
        # self.preds_full_range = self.regressor.predict(self.X_array)
        self.preds_full_range = self.assignment
        if self.verbose: print '[OK] Predictions - %s'%self.assignment_name
        return True
    
    def set_preds_submission(self):
        self.preds_submission = \
        pd.concat([self.preds_full_range \
        .set_index('DATE').loc[self.submission.index] \
        .reset_index(), self.assignment_name_column], axis=1).drop('index', 1)
        return True
        
class Submission:
    def __init__(self,
                 feature_extractor=None,
                 regressor=None,
                 submission_file='./submission.txt',
                 train_dir='./data/CSPL_RECEIVED_CALLS_series/',
                 output_file='./submissions/1.txt',
                 verbose=True):
        
        self.raw_submission = pd.read_csv(submission_file, sep='\t', parse_dates=[0])
        self.submission_file = submission_file
        self.train_dir = train_dir
        self.verbose = verbose
        self.output_file = output_file
        #####
        self.assignments_names = []
        self.assignments = self.init_assignments()
        #####
        
        self.submission_final = self.raw_submission.copy()
    
    def init_assignments(self):
        self.assignments_names = self.raw_submission['ASS_ASSIGNMENT'].unique()
        assignments = OrderedDict()
        for assignment_name in sorted(self.assignments_names):
            assignment = Assignment(assignment_name,
                                    self.submission_file, self.train_dir,
                                    verbose=self.verbose)
            assignments[assignment_name] = assignment
        return assignments
        
    def preprocess(self):
        for assignment_name, assignment in self.assignments.iteritems():
            assignment.preprocess()
        if self.verbose: print '[OK] Preprocessing'
        return True
    
    def transform(self):
        for assignment_name, assignment in self.assignments.iteritems():
            assignment.transform()
        if self.verbose: print '[OK] Feature engineering'
        return True
            
    def predict(self):
        for assignment_name, assignment in self.assignments.iteritems():
            assignment.predict()
            assignment.set_preds_submission()
        if self.verbose: print '[OK] Predictions'
            
    def build_submission_final(self):
        res = []
        for assignment_name, assignment in self.assignments.iteritems():
            res.append(assignment.preds_submission)
        res = pd.concat(res, axis=0).set_index(['DATE', 'ASS_ASSIGNMENT']).sort_index().reset_index()
        self.submission_final = res
        return True
    
    def write_submission(self):
        self.submission_final.to_csv(path_or_buf=self.output_file, sep='\t', index=False)
        return True