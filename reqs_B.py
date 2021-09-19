# coding=utf-8
import os

confidentiality = 0
integrity = 0
availability = 0
authentication = 0
authorization = 0
nonRepudiation = 0
accountability = 0
reliability = 0
privacy = 0
physicalSecurity = 0
forgeryResistance = 0
tamperDetection = 0
dataFreshness = 0
confinement = 0
interoperability = 0
dataOrigin = 0

answerCombinations = []

for q1 in range(1, 7):
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
                                                                    answerCombinations.append((q1, q2, q2_1, q2_2, q2_3,
                                                                                               q2_4, q2_5, q3, q3_1, q4,
                                                                                               q5, q6, q7, q8, q9, q10,
                                                                                               q11))
print len(answerCombinations)
answerCombinations = list(dict.fromkeys(answerCombinations))
print len(answerCombinations)

reqCombinations = []

for answer in answerCombinations:

    if answer[0] == 1:
        confidentiality = 1
        privacy = 1
        integrity = 1
        accountability = 1
        authentication = 1
        authorization = 1
        availability = 1
        reliability = 1
        physicalSecurity = 1

    if answer[0] == 2:
        confidentiality = 1
        privacy = 1
        authentication = 1
        authorization = 1
        integrity = 1
        availability = 1
        reliability = 1
        physicalSecurity = 1
        accountability = 1
        nonRepudiation = 1

    if answer[0] == 3:
        authentication = 1
        authorization = 1
        confidentiality = 1
        integrity = 1
        availability = 1
        reliability = 1
        nonRepudiation = 1
        accountability = 1
        physicalSecurity = 1

    if answer[0] == 4:
        confidentiality = 1
        privacy = 1
        authentication = 1
        authorization = 1
        availability = 1
        reliability = 1
        integrity = 1
        physicalSecurity = 1

    if answer[0] == 5:
        availability = 1
        authentication = 1
        confidentiality = 1
        privacy = 1
        integrity = 1
        tamperDetection = 1

    if answer[0] == 6:
        confidentiality = 1
        integrity = 1
        availability = 1
        authentication = 1
        authorization = 1
        nonRepudiation = 1
        accountability = 1
        reliability = 1
        privacy = 1
        physicalSecurity = 1

    if answer[2] == 1:
        authentication = 1
        nonRepudiation = 1

    if answer[2] == 2:
        authentication = 0
        nonRepudiation = 0

    if answer[3] == 1:
        privacy = 1
        confidentiality = 1

    if answer[3] == 2:
        privacy = 0
        confidentiality = 0

    if answer[4] == 1:
        privacy = 1
        confidentiality = 1

    if answer[4] == 2:
        privacy = 0
        confidentiality = 0

    if answer[5] == 1:
        privacy = 1
        confidentiality = 1
        physicalSecurity = 1

    if answer[5] == 2:
        privacy = 1
        confidentiality = 1
        physicalSecurity = 1
        authorization = 1
        forgeryResistance = 1
        authentication = 1

    if answer[5] == 3:
        privacy = 1
        confidentiality = 1
        physicalSecurity = 1
        authorization = 1
        forgeryResistance = 1
        nonRepudiation = 1
        authentication = 1

    if answer[6] == 1:
        nonRepudiation = 1
        authentication = 1
        confinement = 1

    if answer[7] == 1:
        nonRepudiation = 1
        accountability = 1
        reliability = 1

    if answer[8] == 1:
        integrity = 1
        availability = 1
        dataFreshness = 1
        forgeryResistance = 1
        nonRepudiation = 1
        authentication = 1

    if answer[9] == 1:
        physicalSecurity = 1
        integrity = 1
        availability = 1
        forgeryResistance = 1
        authentication = 1
        nonRepudiation = 1

    if answer[10] == 1:
        availability = 1

    if answer[11] == 1:
        confinement = 1
        interoperability = 1

    if answer[12] == 1:
        authorization = 1

    if answer[13] == 1:
        dataOrigin = 1
        dataFreshness = 1

    if answer[14] == 1:
        authentication = 1

    if answer[15] == 1:
        physicalSecurity = 1

    if answer[16] == 1:
        tamperDetection = 1

    reqList = []

    if confidentiality == 1:
        reqList.append(0)
    if integrity == 1:
        reqList.append(1)
    if availability == 1:
        reqList.append(2)
    if authentication == 1:
        reqList.append(3)
    if authorization == 1:
        reqList.append(4)
    if nonRepudiation == 1:
        reqList.append(5)
    if accountability == 1:
        reqList.append(6)
    if reliability == 1:
        reqList.append(7)
    if privacy == 1:
        reqList.append(8)
    if physicalSecurity == 1:
        reqList.append(9)
    if forgeryResistance == 1:
        reqList.append(10)
    if tamperDetection == 1:
        reqList.append(11)
    if dataFreshness == 1:
        reqList.append(12)
    if confinement == 1:
        reqList.append(13)
    if interoperability == 1:
        reqList.append(14)
    if dataOrigin == 1:
        reqList.append(15)

    s = ", ".join(map(str, reqList))

    reqCombinations.append(s)

i = 0

with open('dataset.txt', 'w') as f2:
    for ans in answerCombinations:
        print >> f2, ans
        print >> f2, reqCombinations[i]
        i += 1
f2.close()

