import os
import json

from django.shortcuts import render
from django.http import JsonResponse

from bsbi import *
from letor import *
from index import *
from compression import *
from util import *
# Create your views here.

def retrieve_serp(request):
    
    if request.method == 'GET':
        response = None
        cur_path = os.path.dirname(__file__)
        with open(os.path.join(cur_path, 'letor.pkl'), 'rb') as file:
            letor = pickle.load(file)
            query = json.loads(request.body)['query']
            response = letor.predict(query)
        
        return JsonResponse(response)