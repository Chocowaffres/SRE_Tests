from os import mkdir, path
from random import uniform
from scipy import sparse
import arff
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from skmultilearn.problem_transform import ClassifierChain, BinaryRelevance
from skmultilearn.ensemble import MajorityVotingClassifier

# -------------------------------------------------------------------------------------

def load_custom_dataset(dataset_name, label_count):
    train_dataset = arff.load(open(dataset_name, 'r'))
    length_train = len(train_dataset['data'])

    X_train = np.array([np.array(train_dataset['data'][i], dtype=float)[:-label_count] for i in range(length_train)])
    Y_train = np.array([np.array(train_dataset['data'][i], dtype=int)[-label_count:] for i in range(length_train)])

    if(length_train != 0):
        X_train = sparse.lil_matrix(X_train, shape=X_train.shape)
        Y_train = sparse.lil_matrix(Y_train, shape=Y_train.shape)

    return X_train, Y_train

def create_dataset_file(full_path):
    f = open(full_path, "w")
    f.write("""@relation MultiLabelData

@attribute Att1 numeric
@attribute Att2 numeric
@attribute Att3 numeric
@attribute Att4 numeric
@attribute Att5 numeric
@attribute Att6 numeric
@attribute Att7 numeric
@attribute Att8 numeric
@attribute Att9 numeric
@attribute Att10 numeric
@attribute Att11 numeric
@attribute Att12 numeric
@attribute Att13 numeric
@attribute Att14 numeric
@attribute Att15 numeric
@attribute Att16 numeric
@attribute Att17 numeric
@attribute Class1 {0,1}
@attribute Class2 {0,1}
@attribute Class3 {0,1}
@attribute Class4 {0,1}
@attribute Class5 {0,1}
@attribute Class6 {0,1}
@attribute Class7 {0,1}
@attribute Class8 {0,1}
@attribute Class9 {0,1}
@attribute Class10 {0,1}
@attribute Class11 {0,1}
@attribute Class12 {0,1}
@attribute Class13 {0,1}
@attribute Class14 {0,1}
@attribute Class15 {0,1}
@attribute Class16 {0,1}

@data
""")
    f.close()
    return
# -------------------------------------------------------------------------------------

def get_question_answer(question_text, question_options, possible_values):
    while True:
        print(question_text)
        print(question_options)
        print("  Answer: \n")

        try:
            answer = int(input())
            if (answer in possible_values):
                return answer
            print("Insert a valid answer \n")
        except:
            print("Insert a valid answer \n")

# -------------------------------------------------------------------------------------

def ask_questions():
    question_text    = "# Question 1 \n## State the domain type for your IoT system: \n"
    question_options = "   1 - Smart Home \n   2 - Smart Healthcare \n   3 - Smart Manufacturing \n   4 - Smart Wearables\n   5 - Smart Toy \n   6 - Smart Transportation\n"
    possible_values  = [1,2,3,4,5,6]

    q1 = get_question_answer(question_text, question_options, possible_values)

    question_text    = "# Question 2 \n## Will the system have a user? \n"
    question_options = "   1 - Yes \n   2 - No \n"
    possible_values  = [1,2]
    
    q2 = get_question_answer(question_text, question_options, possible_values)

    q2_1 = 0
    q2_2 = 0
    if(q2 == 1):
        question_text    = "# Question 2.1 \n## Will the system have user LogIn? \n"
        question_options = "   1 - Yes \n   2 - No \n"
        possible_values  = [1,2]
        
        q2_1 = get_question_answer(question_text, question_options, possible_values)

        question_text    = "# Question 2.2 \n## Will the system hold any user information? \n"
        question_options = "   1 - Yes \n   2 - No \n"
        possible_values  = [1,2]
        
        q2_2 = get_question_answer(question_text, question_options, possible_values)

    q2_3 = 0
    if(q2 == 2 or q2_2 == 2):
        question_text    = "# Question 2.3 \n## Will the system store any kind of information? \n"
        question_options = "   1 - Yes \n   2 - No \n"
        possible_values  = [1,2]
        
        q2_3 = get_question_answer(question_text, question_options, possible_values)

    q2_4 = 0
    q2_5 = 0
    if(q2_2 == 1 or q2_3 == 1):
        question_text    = "# Question 2.4 \n## What will be the level of information stored? \n"
        question_options = "   1 - Normal Information \n   2 - Sensitive Information \n   3 - Critical Information"
        possible_values  = [1,2,3]
        
        q2_4 = get_question_answer(question_text, question_options, possible_values)

        question_text    = "# Question 2.5 \n## Will this information be sent to an entity? \n"
        question_options = "   1 - Yes \n   2 - No \n"
        possible_values  = [1,2]
        
        q2_5 = get_question_answer(question_text, question_options, possible_values)

    question_text    = "# Question 3 \n## Will the system be connected to the internet? \n"
    question_options = "   1 - Yes \n   2 - No \n"
    possible_values  = [1,2]
    
    q3 = get_question_answer(question_text, question_options, possible_values)

    q3_1 = 0
    if(q3 == 1):
        question_text    = "# Question 3.1 \n## Will it send its data to a cloud? \n"
        question_options = "   1 - Yes \n   2 - No \n"
        possible_values  = [1,2]
        
        q3_1 = get_question_answer(question_text, question_options, possible_values)

    question_text    = "# Question 4 \n## Will it store data in a db? \n"
    question_options = "   1 - Yes \n   2 - No \n"
    possible_values  = [1,2]
    
    q4 = get_question_answer(question_text, question_options, possible_values)

    question_text    = "# Question 5 \n## Will the system receive regular updates? \n"
    question_options = "   1 - Yes \n   2 - No \n"
    possible_values  = [1,2]
    
    q5 = get_question_answer(question_text, question_options, possible_values)

    question_text    = "# Question 6 \n## Will the system work with third-party software? \n"
    question_options = "   1 - Yes \n   2 - No \n"
    possible_values  = [1,2]
    
    q6 = get_question_answer(question_text, question_options, possible_values)

    question_text    = "# Question 7 \n## Is there a possibility of the communications being eavesdropped? \n"
    question_options = "   1 - Yes \n   2 - No \n"
    possible_values  = [1,2]
    
    q7 = get_question_answer(question_text, question_options, possible_values)

    question_text    = "# Question 8 \n## Could the messages sent between the system components be captured and resend? \n"
    question_options = "   1 - Yes \n   2 - No \n"
    possible_values  = [1,2]
    
    q8 = get_question_answer(question_text, question_options, possible_values)

    question_text    = "# Question 9 \n## Can someone try to impersonate a user to gain access to private information? \n"
    question_options = "   1 - Yes \n   2 - No \n"
    possible_values  = [1,2]
    
    q9 = get_question_answer(question_text, question_options, possible_values)

    question_text    = "# Question 10 \n## Can someone with bad intentions gain physical access to the location where this software will be running and obtain private information? \n"
    question_options = "   1 - Yes \n   2 - No \n"
    possible_values  = [1,2]
    
    q10 = get_question_answer(question_text, question_options, possible_values)

    question_text    = "# Question 11 \n## Can someone gain physical access to the machine where the system operates or some of the system components and preform some type of modification to it's hardware? \n"
    question_options = "   1 - Yes \n   2 - No \n"
    possible_values  = [1,2]
    
    q11 = get_question_answer(question_text, question_options, possible_values)

    answers = [q1-1, q2-1, q2_1-1, q2_2-1, q2_3-1, q2_4-1, q2_5-1, q3-1, q3_1-1, q4-1, q5-1, q6-1, q7-1, q8-1, q9-1, q10-1, q11-1]

    return answers

def give_security_requirements(answers, root_dir, dataset_file, label_count):
    full_path = root_dir + dataset_file
    sec_reqs  = [0 for _ in range(label_count)]

    # Check if root dir exists
    if (not path.exists(root_dir)):
        mkdir(root_dir)
    
    if (not path.exists(full_path)):
        create_dataset_file(full_path)

    X_train, Y_train = load_custom_dataset(full_path, label_count)

    if (X_train.shape[0] == 0):
        for i in range(len(sec_reqs)):
            monte_carlo = uniform(0,1)
            if monte_carlo < 0.5:
                sec_reqs[i] = 0
            else:
                sec_reqs[i] = 1
    else:
        
        aux_answers = sparse.lil_matrix(np.array(answers))
        cc = ClassifierChain(
            classifier=DecisionTreeClassifier(class_weight='balanced'), 
            require_dense=[False, True],
        )
        sec_reqs = cc.fit(X_train, Y_train).predict(aux_answers).toarray().squeeze()

    return sec_reqs

def validate_security_requirements(sec_reqs, labels_name):
    true_sec_reqs = []

    for i in range(len(labels_name)):
        while True:
            print(labels_name[i] + ":  " + str(sec_reqs[i]))
            try:
                correct = int(input(" Is this label correct? \n 1 - Yes \n 2 - No \n"))
                if correct and correct in [1,2]:
                    break
                print("Insert a valid answer \n")
            except:
                print("Insert a valid answer \n")

        true_sec_reqs.append(sec_reqs[i] if correct == 1 else (0 if sec_reqs[i] == 1 else 1))

    return true_sec_reqs

# -------------------------------------------------------------------------------------

def main():
    root_dir      = 'datasets/'
    dataset_file  = 'continuous_dataset.arff'
    label_count   = 16

    answers       = ask_questions()
    sec_reqs      = give_security_requirements(answers, root_dir, dataset_file, label_count)
    labels_name   = ["Confidentiality", "Integrity", "Availability", "Authentication", "Authorization", "Non-Repudiation", "Accountability", "Reliability", "Privacy", "Physical Security", "Forgery Resistance", "Tamper Detection", "Data Freshness", "Confinement", "Interoperability", "Data Origin"]
    true_sec_reqs = validate_security_requirements(sec_reqs, labels_name)
    
    visual_true_reqs = ""
    visual_sec_reqs  = ""

    for i in range(len(labels_name)):
        if(true_sec_reqs[i] == 1):
            visual_true_reqs += labels_name[i] + "\n"
        if(sec_reqs[i] == 1):
            visual_sec_reqs += labels_name[i] + "\n"

    print("\n\n# Machine given requirements\n")
    print(visual_sec_reqs)

    print("\n\n# Requirements the user considered valid\n")
    print(visual_true_reqs)

    full_path = root_dir + dataset_file
    row       = str(answers + true_sec_reqs)[1:-1] + "\n"

    f         = open(full_path, "a")
    f.write(row)
    f.close()


    return

if __name__ == '__main__':
    main()