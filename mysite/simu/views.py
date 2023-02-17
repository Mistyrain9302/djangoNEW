from django.shortcuts import render


def dist_base(request):
    context = {
        "title" : "분포도"
    }
    return render(request,'simu/dist_base.html',context)

def cluster(request):
    context = {
    "title" : "군집분석"
    }
    return render(request,'simu/cluster.html',context)

def regressor(request):
    context = {
    "title" : "회귀분석"
    }
    return render(request,'simu/regressor.html',context)

def scores(request):
    context = {
        "title" : "환자 점수"
    }
    return render(request,'simu/scores.html',context)
