from django.conf.urls import url
from TWWET_DASHBOARD import views

#for images and static files
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
urlpatterns=[
    url(r'^$',views.index,name='index'),
]
#'$' used for opening index./ once even if clicked twice

#fxn to add path of static files
urlpatterns+=staticfiles_urlpatterns()

#made by own