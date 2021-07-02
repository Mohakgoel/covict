from django.shortcuts import render
from django.http import Http404
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import numpy as np
import json

# Create your views here.

model = load_model('./models/model.h5')

@api_view(["POST"])
def predictImage(request):
    #print('1.....................')
    #print(request)
    #print('2.....................')
    #print(request.POST)
    #print('3.....................')
    #print(request.FILES)
    #print('4.....................')
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName=fs.url(filePathName)

    testimage = '.'+filePathName 
    img = image.load_img(testimage, target_size=(224, 224))
    img = image.img_to_array(img)
    img = img/255
    img = np.expand_dims(img, axis=0)

    pred = model(img).numpy()[0][0]
    if pred <= 1.0 and pred >= 0.5:
        predictedLabel = 'Normal'
    else:
        predictedLabel =  'Covid'   

    return Response(predictedLabel)
