import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from .models import Student, Attendance, CameraConfiguration
from django.core.files.base import ContentFile
from datetime import datetime, timedelta
from django.utils import timezone
import pygame  # Import pygame for playing sounds
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.urls import reverse_lazy
from django.contrib.auth.decorators import login_required
import threading
import time
import base64
from django.db import IntegrityError
from django.contrib.auth.decorators import login_required, user_passes_test
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth import authenticate, login
from django.contrib import messages
from .models import Student

mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def detect_and_encode(image):
    with torch.no_grad():
        boxes, _ = mtcnn.detect(image)
        if boxes is not None:
            faces = []
            for box in boxes:
                face = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                if face.size == 0:
                    continue
                face = cv2.resize(face, (160, 160))
                face = np.transpose(face, (2, 0, 1)).astype(np.float32) / 255.0
                face_tensor = torch.tensor(face).unsqueeze(0)
                encoding = resnet(face_tensor).detach().numpy().flatten()
                faces.append(encoding)
            return faces
    return []

def encode_uploaded_images():
    known_face_encodings = []
    known_face_names = []

    uploaded_images = Student.objects.filter(authorized=True)

    for student in uploaded_images:
        image_path = os.path.join(settings.MEDIA_ROOT, str(student.image))
        known_image = cv2.imread(image_path)
        known_image_rgb = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)
        encodings = detect_and_encode(known_image_rgb)
        if encodings:
            known_face_encodings.extend(encodings)
            known_face_names.append(student.name)

    return known_face_encodings, known_face_names

def recognize_faces(known_encodings, known_names, test_encodings, threshold=0.6):
    recognized_names = []
    for test_encoding in test_encodings:
        distances = np.linalg.norm(known_encodings - test_encoding, axis=1)
        min_distance_idx = np.argmin(distances)
        if distances[min_distance_idx] < threshold:
            recognized_names.append(known_names[min_distance_idx])
        else:
            recognized_names.append('Not Recognized')
    return recognized_names

def capture_student(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        phone_number = request.POST.get('phone_number')
        student_class = request.POST.get('student_class')
        image_data = request.POST.get('image_data')

        if image_data:
            header, encoded = image_data.split(',', 1)
            image_file = ContentFile(base64.b64decode(encoded), name=f"{name}.jpg")

            student = Student(
                name=name,
                email=email,
                phone_number=phone_number,
                student_class=student_class,
                image=image_file,
                authorized=False
            )
            student.save()

            return redirect('selfie_success')  

    return render(request, 'capture_student.html')

def selfie_success(request):
    return render(request, 'selfie_success.html')

def capture_and_recognize(request):
    stop_events = []  
    camera_threads = []  
    camera_windows = [] 
    error_messages = [] 

    def process_frame(cam_config, stop_event):
        """Thread function to capture and process frames for each camera."""
        cap = None
        window_created = False
        try:
         
            if cam_config.camera_source.isdigit():
                cap = cv2.VideoCapture(int(cam_config.camera_source)) 
            else:
                cap = cv2.VideoCapture(cam_config.camera_source)  

            if not cap.isOpened():
                raise Exception(f"Unable to access camera {cam_config.name}.")

            threshold = cam_config.threshold

            pygame.mixer.init()
            success_sound = pygame.mixer.Sound('app1/suc.wav') 

            window_name = f'Face Recognition - {cam_config.name}'
            camera_windows.append(window_name) 

            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print(f"Failed to capture frame for camera: {cam_config.name}")
                    break 
             
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                test_face_encodings = detect_and_encode(frame_rgb)  

                if test_face_encodings:
                    known_face_encodings, known_face_names = encode_uploaded_images()
                    if known_face_encodings:
                        names = recognize_faces(np.array(known_face_encodings), known_face_names, test_face_encodings, threshold)

                        for name, box in zip(names, mtcnn.detect(frame_rgb)[0]):
                            if box is not None:
                                (x1, y1, x2, y2) = map(int, box)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                                if name != 'Not Recognized':
                                    students = Student.objects.filter(name=name)
                                    if students.exists():
                                        student = students.first()

                                        attendance, created = Attendance.objects.get_or_create(student=student, date=datetime.now().date())
                                        if created:
                                            attendance.mark_checked_in()
                                            success_sound.play()
                                            cv2.putText(frame, f"{name}, checked in.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                        else:
                                            if attendance.check_in_time and not attendance.check_out_time:
                                                if timezone.now() >= attendance.check_in_time + timedelta(seconds=60):
                                                    attendance.mark_checked_out()
                                                    success_sound.play()
                                                    cv2.putText(frame, f"{name}, checked out.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                                else:
                                                    cv2.putText(frame, f"{name}, checked in.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                                            elif attendance.check_in_time and attendance.check_out_time:
                                                cv2.putText(frame, f"{name}, checked out.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

          
                if not window_created:
                    cv2.namedWindow(window_name) 
                    window_created = True  
                
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()  
                    break

        except Exception as e:
            print(f"Error in thread for {cam_config.name}: {e}")
            error_messages.append(str(e)) 
        finally:
            if cap is not None:
                cap.release()
            if window_created:
                cv2.destroyWindow(window_name) 

    try:
    
        cam_configs = CameraConfiguration.objects.all()
        if not cam_configs.exists():
            raise Exception("No camera configurations found. Please configure them in the admin panel.")

        for cam_config in cam_configs:
            stop_event = threading.Event()
            stop_events.append(stop_event)

            camera_thread = threading.Thread(target=process_frame, args=(cam_config, stop_event))
            camera_threads.append(camera_thread)
            camera_thread.start()

        while any(thread.is_alive() for thread in camera_threads):
            time.sleep(1)  

    except Exception as e:
        error_messages.append(str(e))
    finally:
        for stop_event in stop_events:
            stop_event.set()

        for window in camera_windows:
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1: 
                cv2.destroyWindow(window)

    if error_messages:
        full_error_message = "\n".join(error_messages)
        return render(request, 'error.html', {'error_message': full_error_message})  # Render the error page with message

    return redirect('student_attendance_list')

def student_attendance_list(request):
    search_query = request.GET.get('search', '')
    date_filter = request.GET.get('attendance_date', '')

    students = Student.objects.all()

    if search_query:
        students = students.filter(name__icontains=search_query)

    student_attendance_data = []

    for student in students:

        attendance_records = Attendance.objects.filter(student=student)

        if date_filter:
            attendance_records = attendance_records.filter(date=date_filter)

        attendance_records = attendance_records.order_by('date')
        
        student_attendance_data.append({
            'student': student,
            'attendance_records': attendance_records
        })

    context = {
        'student_attendance_data': student_attendance_data,
        'search_query': search_query, 
        'date_filter': date_filter       
    }
    return render(request, 'student_attendance_list.html', context)


def home(request):
    total_students = Student.objects.count()
    total_attendance = Attendance.objects.count()
    total_check_ins = Attendance.objects.filter(check_in_time__isnull=False).count()
    total_check_outs = Attendance.objects.filter(check_out_time__isnull=False).count()
    total_cameras = CameraConfiguration.objects.count()

    context = {
        'total_students': total_students,
        'total_attendance': total_attendance,
        'total_check_ins': total_check_ins,
        'total_check_outs': total_check_outs,
        'total_cameras': total_cameras,
    }
    return render(request, 'home.html', context)

def is_admin(user):
    return user.is_superuser

@login_required
@user_passes_test(is_admin)
def student_list(request):
    students = Student.objects.all()
    return render(request, 'student_list.html', {'students': students})

@login_required
@user_passes_test(is_admin)
def student_detail(request, pk):
    student = get_object_or_404(Student, pk=pk)
    return render(request, 'student_detail.html', {'student': student})

@login_required
@user_passes_test(is_admin)
def student_authorize(request, pk):
    student = get_object_or_404(Student, pk=pk)
    
    if request.method == 'POST':
        authorized = request.POST.get('authorized', False)
        student.authorized = bool(authorized)
        student.save()
        return redirect('student-detail', pk=pk)
    
    return render(request, 'student_authorize.html', {'student': student})

@login_required
@user_passes_test(is_admin)
def student_delete(request, pk):
    student = get_object_or_404(Student, pk=pk)
    
    if request.method == 'POST':
        student.delete()
        messages.success(request, 'Student deleted successfully.')
        return redirect('student-list') 
    
    return render(request, 'student_delete_confirm.html', {'student': student})

def user_login(request):

    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect('home')  
        else:
            messages.error(request, 'Invalid username or password.')

    return render(request, 'login.html')

def user_logout(request):
    logout(request)
    return redirect('login') 

@login_required
@user_passes_test(is_admin)
def camera_config_create(request):

    if request.method == "POST":
        name = request.POST.get('name')
        camera_source = request.POST.get('camera_source')
        threshold = request.POST.get('threshold')

        try:
            CameraConfiguration.objects.create(
                name=name,
                camera_source=camera_source,
                threshold=threshold,
            )
            return redirect('camera_config_list')

        except IntegrityError:
            messages.error(request, "A configuration with this name already exists.")
            return render(request, 'camera_config_form.html')

    return render(request, 'camera_config_form.html')

@login_required
@user_passes_test(is_admin)
def camera_config_list(request):
    configs = CameraConfiguration.objects.all()
    return render(request, 'camera_config_list.html', {'configs': configs})

@login_required
@user_passes_test(is_admin)
def camera_config_update(request, pk):
    config = get_object_or_404(CameraConfiguration, pk=pk)

    if request.method == "POST":
        config.name = request.POST.get('name')
        config.camera_source = request.POST.get('camera_source')
        config.threshold = request.POST.get('threshold')
        config.success_sound_path = request.POST.get('success_sound_path')

        config.save()  
    
        return redirect('camera_config_list')  
    
    return render(request, 'camera_config_form.html', {'config': config})

@login_required
@user_passes_test(is_admin)
def camera_config_delete(request, pk):
    config = get_object_or_404(CameraConfiguration, pk=pk)

    if request.method == "POST":
        config.delete()  

        return redirect('camera_config_list')

    return render(request, 'camera_config_delete.html', {'config': config})

LOGIN_URL = '/login/'
LOGIN_REDIRECT_URL = '/home/'
LOGOUT_REDIRECT_URL = '/login/'
