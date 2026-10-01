# Handwritten Notes — Verbatim Transcription (Unchanged Wording)

This file contains the handwritten notes exactly as written in the notebook pages, with no wording,
spelling, grammar or punctuation altered. Pages are grouped by the notebook-photo files so each block
can be cross-verified against the original images.

Source photos referenced: `34860.jpg`, `34861.jpg`, `34862.jpg`, `34863.jpg`, `34864.jpg`,
`34865.jpg`, `34866.jpg`, `34867.jpg`, `34868.jpg`, `34869.jpg`, `34891.jpg`, `34892.jpg`.

---

## Page 1 (photos 34860.jpg, 34861.jpg)

In US / Europe, people does carry guns can have licence, and even in India for big cities while in some other ones people does illegally but for the initiative part of start a well being of a typical city we must as we can save or able to let or decrease the abuse that users/people does in streets, so, in that case with a typical technology as we are in a digital age we can have different type of possible applicable requirement a different, much better set of tools & devices that are able to capture that information of those gun sounds while there are also a part of heavy processing to send the data from some wide spread microcontrollers then speeded it up to the forest & other national scenturies to preserve wildlife as well as life of humans. Why we are concerned as police is a typically a lot powered & helpful way to make far ahead the to the typical way of living in a real world along with technologies. Even with technologies that are present, we applied or use a chainsaw detection & other services also provide, the concept of a smart city aligns with our goal. If people are not co-operating, the systems does this job while some other related issues while present for the sake of help or the workable world. If in India due to heavy crowd & due to more abusive community on some places, its impossible for now but at some states it can be started with those some colleges or those schools institutions we can do use them as expected so that truly work like as we expect. Not only these there are a lot of examples that does this issue while we ensure the goodwill to do it for society. A small idea of this problem & the conclusion of this introduction is as: the gunshots that are

---

## Page 2 (photos 34862.jpg, 34863.jpg, 34864.jpg)

for the society are a typical example of a bad will but with technology not using humans, we detect that these fires it we can easily able to let know where is the event happens while also for the person who gets injured get medical support our law idealogys with the shape of those ways which are like catching criminals + take care of loss caused by artificial weapon to society with the help of tags along with Iot. What we are to do here, we like to do the typical learning run the inference on the Arduino BLE. But with the work and real world problem end a search level thing let us work from way that started like go things natively so we worked like that start with simple run in PC / laptop only after training & then run it on PC -> shifted to the Arduino with models of .h5 to .tflite the we create in it & work on the way. Our goal is to create and run the inference on the Arduino BLE. In this paper, we only worked till like for the typical around of the word & typically we ask it to do that job throy inference & we need to send audio to it just say 1 & 0 to that case. But with this & these action we work around & we tried it on the raspberry pi on the virtual one. Along with it we tried it with 95% less of all these settings while this paper ensures these as future aspect we are working on while our thresholds are still where we are working to. So, I think there are 2 sections I wrote to copy & have much more idea of the implementation & with results, we do have 3 more section left - Implementation -> Evaluating

---

## Page 3 (photos 34865.jpg, 34866.jpg, 34867.jpg)

Result 4 which combines even the data & how it performed on train & validation to check is working of the sheer data.
-> For Implementing we started with cleary of data where all data & all are passed in these, we does cleary using librosa & too much different ways to look along, while cleary was a major task in ML we have taken out data from sources => then we started to trim it along 250ms, 500ms, 1 sec segments for all vids & the try to trim out those ones which are useless to it try small like data set contins like 80K files so, we checked on 1K-3-5K files by manually listen the cycle till we satisfied ourselves that data is cleaned well. Along the way we worked on the supervised models starts with basic on why linear regression for a classification task & also for to create a live system to get some thresholds & give it a try a saw which the best model can be the lowest one so we tried then we get something from it why & how it falls we move to the 2 supervised Models the SVM & Random Forest we tried these models with chages of dataset which gives like 1:1 goes to 10 - gunshot then give it to train & then we started work on it along the way to get same kind of results that gave like 99-100% accuracy on test set with some data we go along the way & still we tried to get more with different combinations with all these combinations, we tried to get results & finally understood the total gunshot capability can be on 1:10 to 1:50 so along the way this could be come a diffrent set of training then once create diffrent trims which trims the data into better segments

---

## Page 4 (photos 34868.jpg, 34869.jpg)

typically this works along the with some more configurations of librosa & we tried to match it with a splitter set which gives up more types of different splitter sounds then we get more different types of values for these tasks & get other fixed kind of sets then we stopped on things & verified it can create dynamic values of trying pipline of those sounds for our purpose. These typical piplins were almost final for us and even trying these with these our newer one symbolic / or supervised learning models SVM / RF but still on real mic it gets bad values. So, we tried recorded results for those & gives us some results so we moved to the CNN models before starting all these we moved to the edge impulse with precessing CNN spectrags ---- some field models to see results but still we saw a lot better accuracy for those models & saw a very bad results for live audio there & then probles are still part of our flow. In the next suite, we saw some other examples to work regard y these fixes but at the end we create a report using it & finally the time is off for these preexisted models & its the on use the CNN models with on our way to try. Then we lookable looked for the CNN architecture which started with 1D CNN & moved to some other models

---

## Page 5 — Left Page (photo 34891.jpg)

2D CNN -> 3D CNN -> enhanced giving some time to these models we got a research paper of the PCEN (Per Channel Energy Normalization). So, what we do tried to work out in a typical way from ways we can & trying to just get some better result but we can see results & also PCEN need diffre kind of cleany of file before touch it.

---

## Page 5 — Right Page (photo 34892.jpg)

trying it for the main model. But In the end I can say that, It worked or not is kinda still a type of mistry & we'll look in the typical way of some deploynt one does gets 100:100 : 100 & test & validation while they on 1:1 in sets of gunshot & non-gunshot for the next - step purpose. When raspberry didn't worked well we have all those results in the way we now get is the final one of all those tried steps. Now talk about something that nobody uses or ever had tried because CNN is everybody does have the PCEN we take from somone as well. So, what unique we used that no one tried in this open standard. [ Voronoi approach does not use any os other than open standards but work & algorithm we just using this for a some CNN model first gives the correct layers like 14 - 16 layers in that paper. This section is for some uniqueness if we does any then we can start with next sub parts. -> The Experimentation I think the most of the part I expained the points structure from start like from the first things Supervised (linear regression) to this virtual raspberry pi testing & mostly I covered. then experimentation & its results which we have a lot of files we do have & some are I didn't put in repo.. so, I don't need to add like its like a 6-paged paper So best thing to do is like keep only the best & the most relevent to working output in paper. For the Results, we finally have the conclusion which consists of our latest execution along with the future aspects + we have tried it finally on the virtual raspberry pi - our next step is typically for the course of work related to the hardw ware as we found some what on raspberry first it does make it perfect & finally go on the arduino nano ble & its our final goal to deploy & that is & that's it on it too.

---

## Conclusion Addition (spoken / dictated, to fold into the paper conclusion)

In conclusion we add how we optimally do this with on-device inference — we can do it in a typical way
to work around with sleeping schedulings of nodes and energy transmission almost 0 by sending, if it is a
gunshot, and send like 1 or 0 as output to sink, and aggregation comes up to become the best part of all
these solutions.
