from django.shortcuts import render, redirect
from django.views import View
from django.http import HttpResponse

#인적정보 컬럼
humancols = ["나이", "몸무게", "성별", "bmi"]
#약정보 컬럼
medcols = ['a-glucosidaseinhibitor', 'dppiv', 'meglitinide', 'metformin', 'sglt2i', 'su', 'tzd', '인슐린종류(속효성사용여부)', '인슐린종류(중간형사용여부)', '인슐린종류(지속형사용여부)', '인슐린종류(초속효성사용여부)', '인슐린종류(혼합형사용여부)']
#검사정보 컬럼
checkcols = ['alt', 'ast','bun', 'cr', 'cr(urine)', 'crp', 'glucose(식전)', 'hba1c', 'hdl',
            'ica', 'ketone(urine)', 'ldl', 'r-gtp', 'tc', 'tg', 'cpep핵의학(식전)', 'cpep핵의학(식후)', 'insulin핵의학(식전)',
            'insulin핵의학(식후)']

def dist_base(request):
    return render(request,'simu/dist_base.html')

def classify(request):
    context={'humancols':humancols,
             'medcols':medcols,
             'checkcols':checkcols}
    return render(request,'simu/classify.html',context)

def regress(request):
    if request.method == "POST":
        print(request)
        humancols_selected = request.POST.getlist('humancols')
        medcols_selected = request.POST.getlist('medcols')
        checkcols_selected = request.POST.getlist('checkcols')
        
        print("POST요청 들어옴")
        print("humancols_list: ",humancols_selected)
        print("medcols_list: ",medcols_selected)
        print("checkcols_list: ",checkcols_selected)
        
        context={'humancols':humancols,
                 'medcols':medcols,
                 'checkcols':checkcols,
                 'humancols_list':humancols_selected,
                 'medcols_list':medcols_selected,
                 'checkcols_list':checkcols_selected,}

        return render(request,'simu/regress.html',context)
    else:
        context={'humancols':humancols,
            'medcols':medcols,
            'checkcols':checkcols,}

        return render(request,'simu/regress.html',context)