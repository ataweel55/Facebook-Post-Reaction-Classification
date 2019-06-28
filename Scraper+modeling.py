
# coding: utf-8

# In[ ]:


##############################################
####################### Scraper
app_id = "1823932391215167"
app_secret = "UsTLZAB7jqx34n8SFEg52RUKEgw" # DO NOT SHARE WITH ANYONE!

access_token = "EAACEdEose0cBALdUdBoKwSoolv1Ucx771yicQn1s4nMebJYyvH6CXYTZC2nZAV5LtXhDhAyGOfBtSfjkIA0rIO00rQxVT6ZAem20DOsZBvv6wmcItU1sklUluhvZAgodq49fndMej9UKAYSaQwYW9q7wnBsbfCHEdA8ZAN09tHKlwM9ZBZBjHjNPSuWZCFqiabiuwCZCfcCXTYXX6djtM3vwEj"


x = ['8860325749','101988643193353','6499393458','119984188013847','7533944086','5953023255','220198801458577','273864989376427','1425464424382692','11643473298','200137333331078','235852889908002','95475020353','15779440092','140738092630206','80256732576','25987609066','43179984254','687156898054966','164305410295882','8304333127','21516776437','10513336322','5863113009','7642602143','268914272540','114114045339706','5281959998','41632789656','695526053890545','10643211755','5863113009','9258148868','40656699159','7629206115','6491828674','13320939444','131010790489','90692553761','85452072376','300341323465160','18468761129','69813760388','6250307292','20324257234','228735667216','144128236879','37763684202','6013004059','338028696036','153005644864469','223649167822693','18343191100','10606591490','62317591679','455410617809655','1416139158459267','13652355666','15704546335','249655421622','182919686769','1481073582140028','102938906478343','114050161948682','98658495398','131459315949','29259828486','147772245840','97212224368','136264019722601','86680728811','60894670532','20446254070']
x2 = ["479042895558058", "89562268312", "6815841748", "81221197163", "153080620724", "4", "197589525931", "11777366210", "104881676217125","247453712038179","11815164971","268201959869854","151000201600745","175462542465962","234202803271464","497509133698345","24154059750","40389960943","10909582943","142427202457055","300224781015","69761195295","26012002239","80500707044","56771207447","8303218826","30776016407","100834231907","58407459601","108755587572","108905269168364","89410513554","123094714264","59306617060","78612416276","1191441824276882","100001760784049","100000282129544"]
for page_id in x:

	def testFacebookPageData(page_id, access_token):
		
		# construct the URL string
		base = "https://graph.facebook.com/v2.4"
		node = "/" + page_id
		parameters = "/?access_token=%s" % access_token
		url = base + node + parameters
		
		# retrieve data
		req = urllib2.Request(url)
		response = urllib2.urlopen(req)
		data = json.loads(response.read())
		
		print (json.dumps(data, indent=4, sort_keys=True))
		

	testFacebookPageData(page_id, access_token)

	def request_until_succeed(url):
		req = urllib2.Request(url)
		success = False
		while success is False:
			try: 
				response = urllib2.urlopen(req)
				if response.getcode() == 200:
					success = True

			except Exception as e:
				print (e)
				time.sleep(5)

				print ("Error for URL %s: %s" % (url, datetime.datetime.now()))

		return response.read()

	# Needed to write tricky unicode correctly to csv
	def unicode_normalize(text):
		return text.translate({ 0x2018:0x27, 0x2019:0x27, 0x201C:0x22, 0x201D:0x22, 0xa0:0x20 }).encode('utf-8')

	def getFacebookPageFeedData(page_id, access_token, num_statuses):
		
		# Construct the URL string; see http://stackoverflow.com/a/37239851 for Reactions parameters
		base = "https://graph.facebook.com/v2.6"
		node = "/%s/posts" % page_id 
		fields = "/?fields=message,link,created_time,type,name,id,comments.limit(0).summary(true),shares,reactions.limit(0).summary(true)"
		parameters = "&limit=%s&access_token=%s" % (num_statuses, access_token)
		url = base + node + fields + parameters
		
		# retrieve data
		data = json.loads(request_until_succeed(url))
		
		return data
		
	def getReactionsForStatus(status_id, access_token):

		# See http://stackoverflow.com/a/37239851 for Reactions parameters
		# Reactions are only accessable at a single-post endpoint
		
		base = "https://graph.facebook.com/v2.6"
		node = "/%s" % status_id
		reactions = "/?fields=" 						"reactions.type(LIKE).limit(0).summary(total_count).as(like)" 						",reactions.type(LOVE).limit(0).summary(total_count).as(love)" 						",reactions.type(WOW).limit(0).summary(total_count).as(wow)" 						",reactions.type(HAHA).limit(0).summary(total_count).as(haha)" 						",reactions.type(SAD).limit(0).summary(total_count).as(sad)" 						",reactions.type(ANGRY).limit(0).summary(total_count).as(angry)"
		parameters = "&access_token=%s" % access_token
		url = base + node + reactions + parameters
		
		# retrieve data
		data = json.loads(request_until_succeed(url))
		
		return data
		

	def processFacebookPageFeedStatus(status, access_token):
		
		# The status is now a Python dictionary, so for top-level items,
		# we can simply call the key.
		
		# Additionally, some items may not always exist,
		# so must check for existence first
		
		status_id = status['id']
		status_message = '' if 'message' not in status.keys() else unicode_normalize(status['message'])
		link_name = '' if 'name' not in status.keys() else unicode_normalize(status['name'])
		status_type = status['type']
		status_link = '' if 'link' not in status.keys() else unicode_normalize(status['link'])
		
		# Time needs special care since a) it's in UTC and
		# b) it's not easy to use in statistical programs.
		
		status_published = datetime.datetime.strptime(status['created_time'],'%Y-%m-%dT%H:%M:%S+0000')
		status_published = status_published + datetime.timedelta(hours=-5) # EST
		status_published = status_published.strftime('%Y-%m-%d %H:%M:%S') # best time format for spreadsheet programs
		
		# Nested items require chaining dictionary keys.
		
		num_reactions = 0 if 'reactions' not in status else status['reactions']['summary']['total_count']
		num_comments = 0 if 'comments' not in status else status['comments']['summary']['total_count']
		num_shares = 0 if 'shares' not in status else status['shares']['count']
		
		# Counts of each reaction separately; good for sentiment
		# Only check for reactions if past date of implementation: http://newsroom.fb.com/news/2016/02/reactions-now-available-globally/
		
		reactions = getReactionsForStatus(status_id, access_token) if status_published > '2016-02-24 00:00:00' else {}
		
		num_likes = 0 if 'like' not in reactions else reactions['like']['summary']['total_count']
		
		# Special case: Set number of Likes to Number of reactions for pre-reaction statuses
		
		num_likes = num_reactions if status_published < '2016-02-24 00:00:00' else num_likes
		
		num_loves = 0 if 'love' not in reactions else reactions['love']['summary']['total_count']
		num_wows = 0 if 'wow' not in reactions else reactions['wow']['summary']['total_count']
		num_hahas = 0 if 'haha' not in reactions else reactions['haha']['summary']['total_count']
		num_sads = 0 if 'sad' not in reactions else reactions['sad']['summary']['total_count']
		num_angrys = 0 if 'angry' not in reactions else reactions['angry']['summary']['total_count']
		
		# Return a tuple of all processed data
		
		return (status_id, status_message, link_name, status_type, status_link,
			status_published, num_reactions, num_comments, num_shares,  num_likes,
			num_loves, num_wows, num_hahas, num_sads, num_angrys)

	def scrapeFacebookPageFeedStatus(page_id, access_token):
		with open('%s_facebook_statuses.csv' % page_id, 'w') as file:
			w = csv.writer(file)
			w.writerow(["status_id", "status_message", "link_name", "status_type", "status_link",
			"status_published", "num_reactions", "num_comments", "num_shares", "num_likes",
			"num_loves", "num_wows", "num_hahas", "num_sads", "num_angrys"])
			
			has_next_page = True
			num_processed = 0   # keep a count on how many we've processed
			scrape_starttime = datetime.datetime.now()
			
			print ("Scraping %s Facebook Page: %s\n" % (page_id, scrape_starttime))
			
			statuses = getFacebookPageFeedData(page_id, access_token, 100)
			
			while has_next_page:
				for status in statuses['data']:
				
					# Ensure it is a status with the expected metadata
					if 'reactions' in status:
						new_status = processFacebookPageFeedStatus(status, access_token)
						
						if new_status[5] < '2017-02-24 00:00:00':
							break
						
						w.writerow(new_status)
					
					# output progress occasionally to make sure code is not stalling
					num_processed += 1
					if num_processed % 100 == 0:
						print ("%s Statuses Processed: %s" % (num_processed, datetime.datetime.now()))

						
				# if there is no next page, we're done.
				if 'paging' in statuses.keys():
					statuses = json.loads(request_until_succeed(statuses['paging']['next']))
				else:
					has_next_page = False
					
			
			print ("\nDone!\n%s Statuses Processed in %s" % (num_processed, datetime.datetime.now() - scrape_starttime))


	if __name__ == '__main__':
		scrapeFacebookPageFeedStatus(page_id, access_token)



##############################################
####################### Modeling - For one label Only, this process is repeated for each label
data = pd.read_csv("NewTrain.csv", encoding="ISO-8859-1")
val = pd.read_csv("NewTest.csv", encoding="ISO-8859-1")
data = data[["post", 'LOVE', 'WOW', 'HAHA', 'SAD','ANGRY']]
data2 = pd.concat([data, data[data.LOVE == 0], data[data.LOVE == 0].sample(26727)])
data2 = data2.sample(frac=1)
data2.index = range(0, len(data2))
data3 = pd.concat([data[data.LOVE == 0], data[data.LOVE == 1].sample(100274)])
data3.index = range(0, len(data3))
data3 = data3.sample(frac=1)
data4 = data.post.str.replace(" ", "")
val4 = val.post.str.replace(" ", "")

d = []
for w in data4:
    x1 = []
    x1.append(" ".join(re.findall("[a-zA-Z]{2}", w))) 
    #x1.append(" ".join(re.findall("[a-zA-Z]{3}", w))) 

    d.append(x1[0])
X_train = pd.Series(d)

for w in val4:
    x1 = []
    x1.append(" ".join(re.findall("[a-zA-Z]{2}", w))) 
    d.append(x1[0])
X_val = pd.Series(d)

data["data4"] = X_train
data4 = data[["data4", 'LOVE', 'WOW', 'HAHA', 'SAD', 'ANGRY']]
data4.columns = ["post", 'LOVE', 'WOW', 'HAHA', 'SAD', 'ANGRY']
data4.to_csv("new\\data4.csv", index = False)

val["val4"] = X_val
val4 = val[["val4", 'LOVE', 'WOW', 'HAHA', 'SAD', 'ANGRY']]
val4.columns = ["post", 'LOVE', 'WOW', 'HAHA', 'SAD', 'ANGRY']
val4.to_csv("new\\val4.csv", index = False, encoding="ISO-8859-1")
temp = pd.concat([data, data[data.LOVE == 0], data[data.LOVE == 0].sample(26727)])
temp = temp.sample(frac = 1)
temp.index = range(0, len(temp))
data5 = temp.post.str.replace(" ", "")
val5 = val.post.str.replace(" ", "")

d = []
for w in data5:
    x1 = []
    x1.append(" ".join(re.findall("[a-zA-Z]{3}", w))) 
    #x1.append(" ".join(re.findall("[a-zA-Z]{3}", w))) 

    d.append(x1[0])
X_train = pd.Series(d)

for w in val5:
    x1 = []
    x1.append(" ".join(re.findall("[a-zA-Z]{3}", w))) 
    d.append(x1[0])
X_val = pd.Series(d)


data2["data5"] = X_train
data5 = data2[["data5", 'LOVE', 'WOW', 'HAHA', 'SAD', 'ANGRY']]
data5.columns = ["post", 'LOVE', 'WOW', 'HAHA', 'SAD', 'ANGRY']
data5.to_csv("new\\data5.csv", index = False)

val["val5"] = X_val
val5 = val[["val5", 'LOVE', 'WOW', 'HAHA', 'SAD', 'ANGRY']]
val5.columns = ["post", 'LOVE', 'WOW', 'HAHA', 'SAD', 'ANGRY']
val5.to_csv("new\\val5.csv", index = False)
temp = pd.concat([data[data.LOVE == 0], data[data.LOVE == 1].sample(100274)])
temp = temp.sample(frac = 1)
temp.index = range(0, len(temp))
data6 = temp.post.str.replace(" ", "")
val6 = val.post.str.replace(" ", "")

d = []
for w in data6:
    x1 = []
    x1.append(" ".join(re.findall("[a-zA-Z]{3}", w))) 
    #x1.append(" ".join(re.findall("[a-zA-Z]{3}", w))) 

    d.append(x1[0])
X_train = pd.Series(d)

for w in val6:
    x1 = []
    x1.append(" ".join(re.findall("[a-zA-Z]{3}", w))) 
    d.append(x1[0])
X_val = pd.Series(d)

data3["data5"] = X_train
data6 = data3[["data5", 'LOVE', 'WOW', 'HAHA', 'SAD', 'ANGRY']]
data6.columns = ["post", 'LOVE', 'WOW', 'HAHA', 'SAD', 'ANGRY']
data6.to_csv("new\\data6.csv", index = False)

val["val5"] = X_val
val6 = val[["val5", 'LOVE', 'WOW', 'HAHA', 'SAD', 'ANGRY']]
val6.columns = ["post", 'LOVE', 'WOW', 'HAHA', 'SAD', 'ANGRY']
val6.to_csv("new\\val6.csv", index = False)
def extra(label,vectorizer, DATA,VAL, NAME, dataNAME, i):
    X_val = VAL["post"]
    y_val = VAL[label]
    t1, t2 = train_test_split(DATA, test_size = 100274, random_state = 10)
    X1 = t1["post"]
    Y1 = t1[label]
    X2 = t2["post"]
    Y2 = t2[label]
    print (X1.shape, " ------------This is T1")
    print (X2.shape, " ------------This is T2")
    print (Y1.shape, " ------------This is Y1")
    print (Y2.shape, " ------------This is Y2")
    vect = vectorizer.fit(X1)
    X_train_vectorized = vect.transform(X1)
    print(X_train_vectorized.shape)
    name = "new\\extra\\train\\" + dataNAME + NAME +"_" + "_" + "X_train_vectorized"    
    X_test_vectorized = vect.transform(X2)
    print(X_test_vectorized.shape)
    name = "new\\extra\\test\\" + dataNAME + NAME +"_" + "_" + "X_test_vectorized"
    X_val_vectorized = vect.transform(X_val)
    print(X_val_vectorized.shape)
    name = "new\\extra\\val\\" +  dataNAME + NAME +"_" +"_" + "X_val_vectorized"
    Y1.to_csv("Y1_"+str(i)+dataNAME+".csv", index = False)
    Y2.to_csv("Y2_"+str(i)+dataNAME+".csv", index = False)
    y_val.to_csv("Y_val_"+str(i)+dataNAME+".csv", index = False)

stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

class StemmedCountVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
    
stemmed_TF_vect = StemmedCountVectorizer(stop_words='english')
vects = {}
vects["BoW"] = CountVectorizer(encoding="ISO-8859-1")
vects["BoWSW"] = CountVectorizer(encoding="ISO-8859-1", stop_words = "english")
vects["BoWSWLower"] = CountVectorizer(encoding="ISO-8859-1", stop_words = "english",lowercase = False )
vects["TF_1_2"] = TfidfVectorizer(ngram_range=(1, 2))
vects["TF_1_3"] = TfidfVectorizer(ngram_range=(1, 3))
vects["TF_1_3_SW"] = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 3), stop_words='english')          
vects["TFStem"] = stemmed_TF_vect

now = datetime.datetime.now()

for i,d in enumerate(["data1.csv", "data2.csv", "data3.csv", "data4.csv", "data5.csv","data6.csv" ]):
    print (d)
    file = pd.read_csv("new\\"+d, encoding = "ISO-8859-1")
    print (file.shape)
    if i+1 >= 4:
        if i+4 == 4:
            VAL = val4
            file = data4
        elif i+4 == 5:
            VAL = val5
            file = data5

        elif i+4 == 6:
            VAL = val6
            file = data6
    
        else:
            print ("error")
    else:
        VAL = val
    for v in vects.keys():
        print (v)
        extra("LOVE",vects[v], file,VAL, v, d.split(".")[0], i)
        print (datetime.datetime.now() - now)
        print ("---------------------")
        
    print ("******************************************************************")
    
print ("TOTAL time",datetime.datetime.now() - now)
def base(MODELS, i):
    models = MODELS

    for m in models.keys():
        print (m)
        model = models[m]
        model.fit(X_train_vectorized, Y1)
        predictions2 = model.predict_proba(X_train_vectorized)
        metaTrain[m+str(i)+"2"] = [p[0] for p in predictions2]   ###############
        predictions = model.predict(X_train_vectorized)
        metaTrain[m+str(i)] = predictions       ###############
        predictions2 = model.predict_proba(X_test_vectorized) 
        metaTest[m+str(i)+"2"] = [p[0] for p in predictions2]  ###############
        predictions = model.predict(X_test_vectorized)
        metaTest[m+str(i)] = predictions       ###############
        print (accuracy_score(Y2, predictions))
        print (f1_score(Y2, predictions, pos_label = 1))
        print (roc_auc_score(Y2, predictions))
        print (classification_report (Y2, predictions))
        print (confusion_matrix (Y2, predictions))
        
        predictions2 = model.predict_proba(X_val_vectorized)
        metaVal[m+str(i)+"2"] = [p[0] for p in predictions2] ###############
        predictions = model.predict(X_val_vectorized)
        metaVal[m+str(i)] = predictions ###############
        print ("**************************************************************")
models = {}
models["RF"] = RandomForestClassifier()
models["LG"] = LogisticRegression()
models["XG"] = XGBClassifier()
models["NB"] = naive_bayes.MultinomialNB()
def meta(MODELS, metaTrain, metaTest, metaVal):
    for m in MODELS.keys():
        print (m)
        model = MODELS[m]
        model.fit(metaTest, Y2)
        predictions = model.predict(metaVal)
        print (accuracy_score(YV, predictions))
        print (f1_score(YV, predictions, pos_label = 1))
        print (roc_auc_score(YV, predictions))
        print (classification_report (YV, predictions))
        print (confusion_matrix (YV, predictions))
        print ("-----------------------------------------------")
        print ("**************************************************************")
models = {}
models["RF"] = RandomForestClassifier()
models["GB"] = GradientBoostingClassifier()
models["LG"] = LogisticRegression()
models["BB"] = naive_bayes.GaussianNB()
models["NN"] = MLPClassifier()
models["SG"] = SGDClassifier()
models["XG"] = XGBClassifier()
models["NB"] = naive_bayes.MultinomialNB()
corr_matrix = metaTest.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
cols = [column for column in upper.columns if any(upper[column] > 0.9)]
metaTest = metaTest[cols]
metaVal = metaVal[cols]
meta(models, metaTrain, metaTest, metaVal)

