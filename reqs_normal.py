# coding=utf-8
import os
import random

answerCombinations = []

for q1 in range(1,7):
    for q2 in range(1, 3):
        for q2_1aux in range(1, 3):
            if q2 == 2:
                q2_1 = -1
            else:
                q2_1 = q2_1aux
            for q2_2aux in range(1, 3):
                if q2 == 2:
                    q2_2 = -1
                else:
                    q2_2 = q2_2aux
                for q2_3aux in range(1, 3):
                    if q2 == 2:
                        q2_3 = -1
                    else:
                        q2_3 = q2_3aux
                    for q2_4aux in range(1, 4):
                        if q2 == 2 or q2_2 == 2 or q2_3 == 2:
                            q2_4 = -1
                        else:
                            q2_4 = q2_4aux
                        for q2_5aux in range(1, 3):
                            if q2 == 2 or q2_2 == 2 or q2_3 == 2:
                                q2_5 = -1
                            else:
                                q2_5 = q2_5aux
                            for q3 in range(1, 3):
                                for q3_1aux in range(1, 3):
                                    if q3 == 2:
                                        q3_1 = -1
                                    else:
                                        q3_1 = q3_1aux
                                    for q4 in range(1, 3):
                                        for q5 in range(1, 3):
                                            for q6 in range(1, 3):
                                                for q7 in range(1, 3):
                                                    for q8 in range(1, 3):
                                                        for q9 in range(1, 3):
                                                            for q10 in range(1, 3):
                                                                for q11 in range(1, 3):
                                                                    answerCombinations.append((float(q1), float(q2), float(q2_1), float(q2_2), float(q2_3),
                                                                                            float(q2_4), float(q2_5), float(q3), float(q3_1), float(q4),
                                                                                            float(q5), float(q6), float(q7), float(q8), float(q9), float(q10),
                                                                                            float(q11)))
print(len(answerCombinations))
random.shuffle(answerCombinations)
answerCombinations = list(dict.fromkeys(answerCombinations))
print(len(answerCombinations))

reqCombinations = []

def initialize_dict():
    dict_ = {
        "confidentiality" : 0,
        "integrity" : 0,
        "availability" : 0, 
        "privacy" : 0,
        "accountability" : 0, 
        "authentication" : 0,
        "authorization" : 0,
        "reliability" : 0,
        "physicalSecurity" : 0, 
        "nonRepudiation" : 0,
        "tamperDetection" : 0, 
        "forgeryResistance" : 0, 
        "confinement" : 0,
        "dataFreshness" : 0, 
        "interoperability" : 0, 
        "dataOrigin" : 0,
    }
    
    return dict_

def insert_security_properties(dict_security_properties, list_security_properties, list_values):
    if len(list_security_properties) != len(list_values):
        raise ValueError("Tamanho das propriedades de segurança é diferente dos valores a atribuir.")

    for i, sec_prop in enumerate(list_security_properties):
        dict_security_properties[sec_prop] = list_values[i]


for answer in answerCombinations:
    properties = initialize_dict()

    if answer[0] == 1:
        sec_props = ["confidentiality", "privacy", "integrity", "accountability", "authentication", "authorization", "availability", "reliability", "physicalSecurity"]
        values = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        insert_security_properties(properties, sec_props, values)

    elif answer[0] == 2:
        sec_props = ["confidentiality", "privacy", "integrity", "accountability", "authentication", "authorization", "availability", "reliability", "physicalSecurity", "nonRepudiation"]
        values = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        insert_security_properties(properties, sec_props, values)

    elif answer[0] == 3:
        sec_props = ["confidentiality", "integrity", "accountability", "authentication", "authorization", "availability", "reliability", "physicalSecurity", "nonRepudiation"]
        values = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        insert_security_properties(properties, sec_props, values)

    elif answer[0] == 4:
        sec_props = ["confidentiality", "privacy", "integrity", "authentication", "authorization", "availability", "reliability", "physicalSecurity"]
        values = [1, 1, 1, 1, 1, 1, 1, 1]
        insert_security_properties(properties, sec_props, values)

    elif answer[0] == 5:
        sec_props = ["confidentiality", "privacy", "integrity", "authentication", "availability", "tamperDetection"]
        values = [1, 1, 1, 1, 1, 1]
        insert_security_properties(properties, sec_props, values)

    elif answer[0] == 6:
        sec_props = ["confidentiality", "privacy", "integrity", "accountability", "authentication", "authorization", "availability", "reliability", "physicalSecurity", "nonRepudiation"]
        values = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        insert_security_properties(properties, sec_props, values)

    if answer[2] == 1:
        sec_props = ["authentication", "nonRepudiation"]
        values = [1, 1]
        insert_security_properties(properties, sec_props, values)

    elif answer[2] == 2:
        sec_props = ["authentication", "nonRepudiation"]
        values = [0, 0]
        insert_security_properties(properties, sec_props, values)

    if answer[3] == 1:
        sec_props = ["privacy", "confidentiality"]
        values = [1, 1]
        insert_security_properties(properties, sec_props, values)

    elif answer[3] == 2:
        sec_props = ["privacy", "confidentiality"]
        values = [0, 0]
        insert_security_properties(properties, sec_props, values)

    if answer[4] == 1:
        sec_props = ["privacy", "confidentiality"]
        values = [1, 1]
        insert_security_properties(properties, sec_props, values)

    elif answer[4] == 2:
        sec_props = ["privacy", "confidentiality"]
        values = [0, 0]
        insert_security_properties(properties, sec_props, values)

    if answer[5] == 1:
        sec_props = ["privacy", "confidentiality", "physicalSecurity"]
        values = [1, 1, 1]
        insert_security_properties(properties, sec_props, values)

    elif answer[5] == 2:
        sec_props = ["privacy", "confidentiality", "physicalSecurity", "authorization", "forgeryResistance", "authentication"]
        values = [1, 1, 1, 1, 1, 1]
        insert_security_properties(properties, sec_props, values)

    elif answer[5] == 3:
        sec_props = ["privacy", "confidentiality", "physicalSecurity", "authorization", "forgeryResistance", "authentication", "nonRepudiation"]
        values = [1, 1, 1, 1, 1, 1, 1]
        insert_security_properties(properties, sec_props, values)

    if answer[6] == 1:
        sec_props = ["nonRepudiation", "authentication", "confinement"]
        values = [1, 1, 1]
        insert_security_properties(properties, sec_props, values)

    if answer[7] == 1:
        sec_props = ["nonRepudiation", "accountability", "reliability"]
        values = [1, 1, 1]
        insert_security_properties(properties, sec_props, values)

    if answer[8] == 1:
        sec_props = ["integrity", "availability", "dataFreshness", "forgeryResistance", "nonRepudiation", "authentication"]
        values = [1, 1, 1, 1, 1, 1]
        insert_security_properties(properties, sec_props, values)

    if answer[9] == 1:
        sec_props = ["physicalSecurity", "integrity", "availability", "forgeryResistance", "authentication", "nonRepudiation"]
        values = [1, 1, 1, 1, 1, 1]
        insert_security_properties(properties, sec_props, values)

    if answer[10] == 1:
        sec_props = ["availability"]
        values = [1]
        insert_security_properties(properties, sec_props, values)

    if answer[11] == 1:
        sec_props = ["confinement", "interoperability"]
        values = [1, 1]
        insert_security_properties(properties, sec_props, values)

    if answer[12] == 1:
        sec_props = ["authorization"]
        values = [1]
        insert_security_properties(properties, sec_props, values)

    if answer[13] == 1:
        sec_props = ["dataOrigin", "dataFreshness"]
        values = [1, 1]
        insert_security_properties(properties, sec_props, values)

    if answer[14] == 1:
        sec_props = ["authentication"]
        values = [1]
        insert_security_properties(properties, sec_props, values)

    if answer[15] == 1:
        sec_props = ["physicalSecurity"]
        values = [1]
        insert_security_properties(properties, sec_props, values)

    if answer[16] == 1:
        sec_props = ["tamperDetection"]
        values = [1]
        insert_security_properties(properties, sec_props, values)

    reqList = [0 for i in range(0,16)]

    if properties["confidentiality"] == 1:
        reqList[0] = 1
    if properties["integrity"] == 1:
       reqList[1] = 1
    if properties["availability"] == 1:
       reqList[2] = 1
    if properties["authentication"] == 1:
        reqList[3] = 1
    if properties["authorization"] == 1:
       reqList[4] = 1
    if properties["nonRepudiation"] == 1:
        reqList[5] = 1
    if properties["accountability"] == 1:
       reqList[6] = 1
    if properties["reliability"] == 1:
        reqList[7] = 1
    if properties["privacy"] == 1:
       reqList[8] = 1
    if properties["physicalSecurity"] == 1:
        reqList[9] = 1
    if properties["forgeryResistance"] == 1:
       reqList[10] = 1
    if properties["tamperDetection"] == 1:
       reqList[11] = 1
    if properties["dataFreshness"] == 1:
       reqList[12] = 1
    if properties["confinement"] == 1:
       reqList[13] = 1
    if properties["interoperability"] == 1:
        reqList[14] = 1
    if properties["dataOrigin"] == 1:
        reqList[15] = 1

    s = ", ".join(map(str, reqList))
    reqCombinations.append(s)

i = 0

f1 = open('dataset_normal_train.csv', 'w')
f2 = open('dataset_normal_test.csv', 'w')

for ans in answerCombinations:
    monte_carlo = random.uniform(0,1)
    s = ", ".join(map(str, list(ans)))
    row = str(s) + ", " + str(reqCombinations[i]) + "\n"
    if monte_carlo <= 0.20:
        f2.write(row)
    else:
        f1.write(row)
    i += 1

f1.close()
f2.close()

