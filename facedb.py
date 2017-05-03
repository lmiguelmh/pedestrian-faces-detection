"""
@author lmiguelmh
@since 20170503
"""

import face_recognition
import os

base = "/home/deeplearning/Desktop/projects/tesis/datasets/lfw/lfw-deepfunneled/"
people = ["Aaron_Peirsol", "Aaron_Sorkin", "Abdel_Nasser_Assidi", "Abdullah", "Abel_Pacheco", "Adam_Sandler",
          "Adam_Scott", "Adolfo_Aguilar_Zinser", "Ahmed_Chalabi", "Ai_Sugiyama", "Aicha_El_Ouafi",
          "Akbar_Hashemi_Rafsanjani", "Akhmed_Zakayev", "Al_Gore", "Alan_Ball", "Alberto_Fujimori",
          "Alberto_Ruiz_Gallardon", "Albrecht_Mentz", "Alec_Baldwin"]


def enroll(directory, people):
    templates = []
    names = []
    for person in people:
        path = directory + person + "/" + person + "_0001.jpg"
        # print(path)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) == 1:
            templates.append(encodings[0])
            names.append(person)
        else:
            print("0 or 2+ faces detected in ", person)
    return names, templates


def get_candidates_names(template, names, templates, tolerance=0.6):
    results = face_recognition.compare_faces(templates, template, tolerance)
    candidates_names = []
    for i, result in enumerate(results):
        if result:
            candidates_names.append(names[i])
    return candidates_names