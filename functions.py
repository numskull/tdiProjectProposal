import pandas as pd
import numpy as np
from numpy import dot, transpose
from numpy import dot, transpose
from pathlib import Path
from scipy.sparse import lil_matrix, save_npz, load_npz
from scipy.sparse.linalg import svds
import os
import re
from nltk.corpus import stopwords
import csv
import string
from math import degrees
import networkx as nx

def cleanTweets(com):
    print(f'Cleaning terms of length: {len(com)}')
    stops = stopwords.words("english")
    cleanCom = re.sub(r"(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)", '', com)
    table = str.maketrans('', '', string.punctuation)
    cleanCom = cleanCom.translate(table)
    table = str.maketrans('', '', string.digits)
    cleanCom = cleanCom.translate(table)
    cleanCom = re.sub(r'http\S+', '', cleanCom).split(" ")
    cleanCom = sorted([w for w in cleanCom if w.lower() not in stops])
    print("cleaned.")
    return(cleanCom)

if not os.path.isfile("lang_filter_reduced.csv"):
    data_path = os.path.join(Path(__file__).parents[1], "data", "ira_tweets_csv_hashed.csv")
    print(data_path)
    data = pd.read_csv(data_path)
    print(data.head())
    lang = data[['tweet_language']]
    langIndex = data.loc[data.tweet_language == 'en']
    #langIndex.to_csv("lang_filter_2.csv", sep=',')
    langIndex.to_csv("lang_filter_reduced.csv", sep=',')
    print(langIndex[1:10])
elif not os.path.isfile("terms.csv"):
    data = pd.read_csv("lang_filter_reduced.csv", encoding="latin")
    combinedTweets = pd.Series(data.tweet_text).str.cat(sep=" ").lower()
    terms = cleanTweets(combinedTweets)
    terms = pd.DataFrame(data=np.array(terms))[0].unique()
    print(type(terms))
    np.savetxt("terms.csv", terms, delimiter=',', fmt="%s", encoding="latin")
    print(terms)
elif not os.path.isfile("author_terms_matrix.npz"):
    data = pd.read_csv("lang_filter_reduced.csv", encoding="latin")
    with open('terms.csv', 'r') as f:
        r = csv.reader(f)
        terms = list(r)
    users = data.userid.unique()
    auth_terms = lil_matrix((len(terms),len(users)))

    for i in range(0, len(users)):
        thisId = users[i]
        print(f'Current ID: {thisId}')
        tweets = data.loc[data.userid == thisId].tweet_text
        compiled = cleanTweets(pd.Series(tweets).str.cat(sep=" ").lower())
        vec = pd.DataFrame(compiled)[0].value_counts().to_dict()
        print(vec)
        values = {}
        print(type(vec))
        print(f'terms type: {type(terms)} and values: {type(str(terms[1]))}')
        for term in terms:
            term = str(term)
            table = str.maketrans('', '', string.punctuation)
            term = term.translate(table)
            table = str.maketrans('', '','"')
            term = term.translate(table)
            table = str.maketrans('', '', "'")
            term = term.translate(table)
            #print(f'Term type: {type(term)} and value: {term}')
            if(term in vec.keys()):
               # print(f'{term} found in vec.keys()')
                values[term] = vec[term]
            else:
                values[term] = 0
        auth_terms[:,i] = np.array(list(values.values()))
        #vec = vec.loc[vec.index.isin(terms)]
        #vec.to_csv('vec.csv', sep=',')
        #vec = [compiled.count(x) for x in terms]
    save_npz("author_terms_matrix.npz", auth_terms.tocsr())
elif not os.path.isfile("users.csv"):
    data = pd.read_csv("lang_filter_reduced.csv", encoding="latin")
    userids = data.userid.unique()
    users = {}
    for user in userids:
        print(f"UserID: {user}")
        sn = data.loc[data.userid == user].user_screen_name.unique()[0]
        print(f'found pair: {user}, {sn}')
        users[user] = sn

    with open('users.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["userid","screenName"])
        for key,value in users.items():
            writer.writerow([key,value])


else:
    #data = pd.read_csv("lang_filter_reduced.csv", encoding="latin")
    mat = load_npz("author_terms_matrix.npz")
    mat = transpose(mat)
    u,s,vt = svds(mat)
    np.savetxt('u.csv', u, delimiter=',')
    np.savetxt("v.csv", vt, delimiter=',')
    np.savetxt("s.csv", s, delimiter=',')


def wordToWord(u, s, termOne, termTwo, columns):
    termOne = columns.index([termOne])
    termTwo = columns.index([termTwo])
    us = dot(u, np.diag(s))
    dist = cosine(us[termOne,], us[termTwo,])
    return(dist)
    #auth_terms = pd.DataFrame(auth_terms, index=data.userid, columns=terms)

def mostSimilarTerms(u,s, term, columns, numTerms):
    term = columns.index([term])
    us = dot(u,np.diag(s))
    values = {}
    for i in range(0, u.shape[0]):
        dist = 0
        if(i != term):
            dist = cosine(us[term,], us[i,])
            if dist > 0 :
                values[columns[i][0]] = dist
    values = sorted(values.items(), key=lambda kv: kv[1])[0:numTerms]
    return values

def userToUser(v,s,userOne,userTwo, users=pd.read_csv("users.csv")):
    userOne = users.loc[users.screenName == userOne].index[0]
    userTwo = users.loc[users.screenName == userTwo].index[0]
    # need to transpose v to get the v matrix and do the dot product since v is already transposed from SVD.
    vs = dot(transpose(v),np.diag(s))
    return(cosine(vs[userOne,], vs[userTwo,]))

def userToAll(v,s,user, numUsers, users = pd.read_csv("users.csv")):
    user = users.loc[users.screenName == user].index[0]
    values = {}
    vs = dot(transpose(v),np.diag(s))
    for i in range(0, v.shape[1]):
        dist = 0
        if(i != user):
            dist = cosine(vs[user,], vs[i,])
            if dist > 0 :
                values[users.screenName[i]] = dist
    values = sorted(values.items(), key=lambda kv: kv[1])[0:numUsers]
    return(values)

def wordToUser(u,s,v, term, user, columns, users = pd.read_csv("users.csv")):
    v = transpose(v)
    term = columns.index([term])
    user = users.loc[users.screenName == user].index[0]
    us = dot(u, np.sqrt(np.diag(s)))
    vs = dot(v, np.sqrt(np.diag(s)))
    dist = cosine(us[term,], vs[user,])
    return(dist)

def wordToAllUsers(u,s,v, term, columns, numUsers, users = pd.read_csv("users.csv")):
    v = transpose(v)
    us = dot(u, np.sqrt(np.diag(s)))
    vs = dot(v, np.sqrt(np.diag(s)))
    term = columns.index([term])
    values = {}
    for user in users.screenName:
        sn = user
        user = users.loc[users.screenName == user].index[0]
        dist = cosine(us[term,], vs[user,])
        if(dist > 0):
            values[sn] = dist
    values = sorted(values.items(), key=lambda kv: kv[1])[0:numUsers]
    return(values)

def userToWords(u,s,v, user, columns, numWords, users = pd.read_csv("users.csv")):
    v = transpose(v)
    us = dot(u, np.sqrt(np.diag(s)))
    vs = dot(v, np.sqrt(np.diag(s)))
    user = users.loc[users.screenName == user].index[0]
    values = {}
    for term in columns:
        if(term != ' ' or term != ''):
            i = columns.index([term])
            dist = cosine(us[i,], vs[user])
            if(dist > 0):
                values[columns[i][0]] = dist
    values = sorted(values.items(), key=lambda kv: kv[1])[0:numWords]
    return(values)

def buildNetwork(user, numUsers, v=np.genfromtxt("v.csv", delimiter=','),
                 s=np.genfromtxt("s.csv", delimiter=','), users = pd.read_csv("users.csv"),
                 network = nx.Graph()):
    sn = user
    nodes = userToAll(v,s, user, numUsers, users)
    for edge in nodes:
        network.add_edge(sn, edge[0], weight=edge[1])
        secondaryNodes = userToAll(v,s, edge[0], numUsers, users)
        for sec in secondaryNodes:
            network.add_edge(edge[0], sec[0], weight=sec[1])
    return(network)

def buildWordNetwork(term, numUsers,columns, u=np.genfromtxt("u.csv", delimiter=','),
                     v=np.genfromtxt("v.csv", delimiter=','),
                    s=np.genfromtxt("s.csv", delimiter=','),
                    users = pd.read_csv("users.csv"),
                    network = nx.Graph()):
    nodes = wordToAllUsers(u,s,v, term, columns, numUsers, users)
    sn = nodes[0][0]
    for edge in nodes:
        network.add_edge(sn, edge[0], weight=edge[1])
        secondaryNodes = userToAll(v,s, edge[0], numUsers, users)
        for sec in secondaryNodes:
            network.add_edge(edge[0], sec[0], weight=sec[1])
    return(network)

def get_centrality(n):
    return(sorted(nx.communicability_betweenness_centrality(n).items(), key=lambda kv: kv[1]))

def cosine(x,y):
    top = dot(x,y)
    bottom = np.sqrt((x*x).sum()) * np.sqrt((y*y).sum())
    cos = top/bottom
    acos = degrees(np.arccos(cos))
    return(acos)

