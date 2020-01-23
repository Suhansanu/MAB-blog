# Hierarchical Multi-Armed Bandits for Discovering Hidden Populations

With the digitalization of a large fraction of the world, public information is now accessible and avaiable from online social networks. Social experiments such as [Milgram's experiment](https://en.wikipedia.org/wiki/Milgram_experiment) and [mental health related experiments](https://www.aaai.org/ocs/index.php/ICWSM/ICWSM13/paper/viewFile/6124/6351) can now be efficiently be performed on online social networks. Interestingly, social scientists are increasingly interested in understanding the online behavior of the hidden populations such as people with mental illnesses, sex workers and paid posters. Furthermore, businesses are interested in advertising their products selectively to specific groups of individuals. In this work, we shall see how these individuals with hidden property that cannot be directly queried via online interfaces can be sampled. 

The problem of sampling hidden population sampling is hard due to a number of reasons. One, the online social networks such as Facebook and Twitter are vast having billions of users. We are interested in only a specific subset of population, thus it resembles as the problem of searching *a needle in haystack*. Secondly, we are limited by the application programming interfaces that permits only a limited number of queries to be made. Thus, we need to define efficient samplers that can yield a large fraction of these online hidden populations. 



## Related work

There are several potential strategies to search for hidden populations on a social network. One strategy is to exploit
the graph structure as in [Respondent Driven Sampling](http://www.respondentdrivensampling.org/reports/RDS1.pdf) or [web-crawling](https://www.sciencedirect.com/science/article/pii/S1389128699000523). At a high-level, a key limitation of graph-based navigation strategy is that the local graph structure can limit our efforts to traverse the entire graph. In
contrast, we can use the social network API to query entities using content (entity attributes) directly; the resulting entities that satisfy the query may be present anywhere on the social graph. One could also view the problem as reconstructing the underlying entity database of the social network. Unlike [database reconstruction problem](https://arxiv.org/abs/1208.0075), our problem is much more restricted--we aim to obtain only a subset of the database. Query reformulation is another promising approach. [Query reformulation systems](https://www.sciencedirect.com/science/article/pii/S030645730500066X) typically use query log data to rewrite a query to maximize the number of relevant documents returned, where relevance is typically computed using the similarity of the query to the document. However, hidden properties are not directly accessible from the document text, making query reformulation challenging.

In this work, we shall use the public application interface, specifically the attributed search, to sample hidden populations. 

## Proposed Sampling Design 

Before we delve into the design of the sampler, we first describe the high level sampling framework through a representative example. 

Consider a healthcare expert is interested in using Twitter's [advanced search interface](https://twitter.com/search-advanced?lang=en) to understand the behavior of a individuals that have some health issues (hidden property). The researcher uses a *classifier or an expert* to classify whether a sampled user belongs to the hidden population or not. The researcher uses the advanced search attributes like language, location, dates, and keywords to sample Twitter users. As shown below, there are several possible queries that can be formed using these attributes as shown below. 

| hashtag | location  | date | 
| ------- | --------- | ---- |
| #Cubs   | Chicago   |  Jan |
| #Cubs   | Chicago   |  Jan |
|...|
|#Cubs    | New York  | Jan  |
|#Dodgers | Los Angeles| March|
|#Yankees | New York   | Jan  |

We notice that there are combinatorial possible queries that can be formed using just few attributes. Given the limited number of API queries available, it is imperative that we design efficient sampling strategy that can quickly find the effective queries that will yield high number of hidden population individuals. 

Next, we observe that Twitter's Application Programming Interface (API) acts as a black-box and may return different number of individuals from hidden population for the same query in different pages, may return the same individual for different queries, and may return fewer than expected or no individuals for some queries. The following figure shows a typical API that returns five individuals for a given query, where the hidden population entities are represented in black.

<img src="/img/blackbox_results.png" width="600" height="250">


We address the problem of combinatorial search space by hierarchically organizing the query space in the form of a tree. Then, we use a decision-tree based search strategy that exploits the correlation between queryable attributes and hidden property to systematically explore the query space by expanding along high yielding decision-tree branches. To this effect, we propose a new attributed search based sampler DT-TMP that combines decision tree and thompson sampling to efficiently sample hidden populations from online social networks. Next, we describe the Thompson sampler followed by decision tree Thompson sampler. 


### Thompson (TMP) algorithm 

[Thompson sampling](https://en.wikipedia.org/wiki/Thompson_sampling) is a heuristic for choosing queries that addresses the exploration and exploitation tradeoff. Given a set of queries, ![$q_1, q_2, \dots q_k$](https://render.githubusercontent.com/render/math?math=%24q_1%2C%20q_2%2C%20%5Cdots%20q_k%24)
, it estimates the expected reward of observing hidden population entities on issuing each query. Based on the estimate, Thompson sampling chooses the query that maximizes the expected reward with respect to a randomly drawn belief. We estimate the reward of a query using the following equation which takes into account the API's black-box issues discussed above. 

![equation](https://render.githubusercontent.com/render/math?math=%5Cmathbb%7BE%7D%5Br_%7Bq%7D%5D%20%3D%20%5CBig(%20%5Cunderbrace%7B%5Cfrac%7BS_q%7D%7BS_q%20%2B%20F_q%7D%7D_%5Ctext%7Bexpected%20%5C%23%20targets%7D%20%5Ccdot%20%5Cunderbrace%7B%5Cfrac%7BN_q-n_q%7D%7BN_q%7D%7D_%5Ctext%7Bnew%7D%20%5Ccdot%20%5Cunderbrace%7B%5CBig(%201-%20%5Cbig(1%20-%20%5Cfrac%7B1%7D%7BN_q%7D%20%5Cbig)%5E%7Bm%7D%7D_%5Ctext%7Bunique%7D%20%5CBig))



### DT-TMP (Decision Tree Thompson) algorithm 

![model](/img/model.png)

We show the workings of the DT-TMP algorithm through an illustrative example shown above. For hidden population of 'mental illness' (represented in red color), the DT-TMP searches the population for the best combinatorial query comprising of two queryable attributes: income and age. It first uses <*, *> query to find the best single attributed query from queries such as <Low, *> and <*, Young>. Subsequently, it finds the best query <Low, *> along which it expands its query search. The decision tree on the right shows the query expansion with the query expansion along green links. 

In summary, DT-TMP mantains an estimate of discovering number of hidden population entities for each query. It uses a [Thompson sampling](https://link.springer.com/chapter/10.1007/978-3-642-34106-9_18) framework to address the exploration and exploitation tradeoff. Furthermore, DT-TMP organizes the query in a tree fashion where general queries are near the root and specific queries are near the leaf. This organization allows DT-TMP to reject poor yielding branches or poor yielding queries (exploiting the correlation between attributes and hidden property) and greedily exploit the high yielding branches. 




## How does DT-TMP perform in practice?

We test our algorithm against standard state-of-the-art hidden population samplers on three offline and three online datasets for a multitude of twelve sampling tasks. The following table illustrates the sampling performance of DT-TMP across several tasks (example, T1: female Twitter users, T2: Twitter users having verified account, T4: doctors rated 5-star on RateMD). 

![model](/img/results.png)


## Future directions

While a number of previous studies have focused on specific models for sampling, for example sampling using graph APIs or sampling using attributed search or sampling through keyword based search, we need to develop models that can use richer query models that exploits the usefulness of different models while considering the sampling cost tradeoffs. Nazi et al.'s [work](https://www.aaai.org/ocs/index.php/ICWSM/ICWSM15/paper/view/10577) is one of the poineering works in this direction. Furthermore, real world samplers suffer with the problem of inaccuracy from the classifiers, and missing and noisy information. A more robust sampling methodology is needed that can address the above problems. Furthermore, we could use the attributed search to not just sample hidden population nodes efficiently but also estimate properties of interest like distribution and correlation metrics in a sample efficient manner. A theoretical understanding of the problem hardness is currently missing. 

## Further Information

This blog is based on the paper, 'Hierarchical Multi-Armed Bandits for Discovering Hidden Populations',

    @inproceedings{kumar2019hierarchical,
        title        = {Hierarchical multi-armed bandits for discovering hidden populations},
        author       = {Kumar, Suhansanu and Gao, Heting and Wang, Changyu and Chang, Kevin Chen-Chuan and Sundaram, Hari},
        booktitle    = {Proceedings of the 2019 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining},
        pages        = {145--153},
        year         = {2019}
        }


The [paper](https://asonamdata.com/ASONAM2019_Proceedings/pdf/papers/021_0145_023.pdf) and [code](/code/) are available. 
