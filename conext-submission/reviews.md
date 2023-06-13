CoNEXT 2022 Paper #171 Reviews and Comments
===========================================================================
Paper #171 Estimating WebRTC-based Video QoE Metrics using Passive Network
Measurements


Review #171A
===========================================================================

Overall merit
-------------
3. Weak accept

Reviewer expertise
------------------
2. Some familiarity

Paper summary
-------------
The paper presents a technique to estimate the video bitrate, FPS and jitter from RTP traffic traces. The paper proposes a simple technique to extract these features from RTP traffic or UDP/IP packets (in cases where the payload is not visible). The paper evaluates the technique in an emulated setup.

Reasons to Accept
-----------------
The paper is very smoothly written. It presents a simple technique that shows good results. The technique may be beneficial to operators who wish to better understand application-layer metrics for video conferencing streams.

Reasons to Reject
-----------------
The evaluation avoids certain key considerations such as packet loss. Thus, it is not clear if the accuracy results would generalize to other scenarios.

Comments for author
-------------------
The paper presents a technique to estimate the video bitrate, FPS and jitter from RTP traffic traces. The paper proposes a simple technique to extract these features from RTP traffic or UDP/IP packets (in cases where the payload is not visible). The paper evaluates the technique in an emulated setup. The technique is simple but exhibits good results. My main suggestion would be to expand the evaluation - the current results are convincing but based on relatively static/simple network conditions, side stepping complexities such as packet loss. This makes it hard to discern how well the techniques would work in the wild. There are also limited insights into the practical challenges faced when trying to process this data at line-rate without packet sampling. That said, overall, the paper is relatively comprehensive and it very clearly written, so kudos to the authors. I have included further thoughts which I hope can be helpful in future work. 

* The approach presented in Section 3.2 seems very sensible and intuitive. As a research contribution, it would be nice to include a few details about potential challenges here. The design of RTP makes these estimations quite straightforward, but I was wondering if there might be some interesting implementation challenges here. For example, a lot of traffic traces are heavily sampled (for obvious reasons), which may prove challenging for identifying things like frame boundaries. It would be nice to include some of these practical issues. 

* You state in the introduction that past work also assumes operators can access application-level headers" as critique - would be good to include references to what you're thinking of here.

* I may has missed this, but what do the percentages in Figure 1 indicate?

* For ET_i, I assume this is the timestamp of the last packet recorded in the trace, rather than the RTP timestamp? 

* You say that in Figure 1 for frames with more than two packets, you only show the max size difference. Why was this? Intuitively, I would have thought it'd be more important to look at the minimum difference (as this would be the harder one to differentiate automatically)?

* It was not clear what you testbed consisted of and how you implemented your traffic shaping based on the MLab traces. Adding some extra information here would be helpful. How do you shape the downlink on endpoint B? Are the two endpoints on two different physical machines? How are they connected? What was the distribution of bandwidth/RTT values you took from the MLab dataset? 

* When emulating the bandwidth, am I right in thinking that you select a new bandwidth every 1 second within a given experiment? So, a given stream would experience a potential bandwidth change every 1 second?

* I believe both Meet and Teams employ ABR to adapt to network conditions. From your later results, it is clear that different frames employ a variety of encoding rates. Can you estimate to what extent this is driven by the underlying video encoding vs network conditions. Does ABR (to reflect varying network conditions) impact your technique for encoding rate estimation? 

* The frame estimation for UDP traffic seems to be impacted by packet reordering. Could you do a quick test to understand how this impacts your estimates? 

* In Figure 5b, am I right in thinking that the high errors mentioned are primarily caused by the occasional spikes? Or is the high error rate caused by consistent errors - the viz makes it a little hard to identify this.

* I would say that the main limitation of the evaluation is that it currently does not dissect fluctuations in the network. For example, variations in bandwidth may trigger changes in the ABR encoding; path issues may result in reordering; packet loss may result in a subset of a frame being delivered. It seems likely these would have a notable impact on the accuracy.

* Typo: "network operators do not have access to RTP headers as these VCAs." - I assume this sentence needs completing?

* Typo: in the Matching ground truth with estimates paragraph, you have a broken Figure \ref



Review #171B
===========================================================================

Overall merit
-------------
3. Weak accept

Reviewer expertise
------------------
3. Knowledgeable

Paper summary
-------------
This paper proposes an approach for network operators to passively measure the QoE of video conferencing applications by leveraging IP/UDP headers. The approach uses properties of different transmissions types (packet sizes) to isolate traffic containing video packets and then uses the header information and the packet size to infer three metrics including (i) bitrate, frame-rate and frame jitter. Evaluation against ground truth data shows that the approach allows estimation of these three metrics with reasonable accuracy.

Reasons to Accept
-----------------
- Passive measurements make the approach theoretically deployable for network operators.
- Video conferencing applications drive significant fraction of Internet traffic, so increased visibility in this traffic is invaluable.

Reasons to Reject
-----------------
- Doesn't directly evaluate the precision/recall in detecting frame boundaries which is of central importance to the paper.
- Paper lacks sufficient insights into observed behavior and analysis is often coarse-grained.
- Paper has missing and wrongly referenced graphs which makes it hard to review.

Comments for author
-------------------
Thanks for submitting this paper, I liked the motivation -- any visibility in VCA traffic is useful for network operators and usage of passive measurements potentially makes the technique deployable.

My main concern is that while the paper correctly realizes the central challenge of accurately identifying frame boundaries, it doesn't do justice in terms of evaluating how effectively it overcomes this challenge. For instance, it isn't sufficient to claim that frame boundary detection is accurate based on Figure 4 alone. It would be better to show the percentage of actual frames which were detected correctly. Figure 5a (wrongly referenced in text as 6a) helps to show a good case, but are there cases where the approach struggled to accurately detect frames and why?

Similarly, the comparison with ground truth is done over per-second aggregated metrics. This is a mismatch, if the goal is to detect frame-boundaries, then metrics such as bitrate and frame jitter should be compared at per-frame level. It may be possible to augment WebRTC to report per frame statistics (including render, decode, jitter buffer delay), see this paper for reference: [Dissecting Cloud Gaming Performance with DECAF](https://dl.acm.org/doi/abs/10.1145/3489048.3522628).

I also think that paper should provide deeper insights in the results. For example, while discussing Figure 2, the paper states that FEC and padding overheads cause overestimation of bitrate in 80% of the cases, however, it isn’t clear what causes underestimation of bitrate in the remaining 20%. The paper doesn’t discuss underestimation at all. Furthermore, I believe it is possible to obtain total bytes and media payload separately in WebRTC which can help in verifying the reasons given for over-estimation ([this may be relevant](https://chromium.googlesource.com/external/webrtc/+/HEAD/video/receive_statistics_proxy.cc#416)).

Finally, the paper contains a large number of typos, missing and wrongly referenced figures which made it very hard to review. Following is a list of issues which I noted.

* Sec 2.2 Problem Statement: “We take as input … and outputs the desired QoE” -> and **output** the desired QoE
* Sec 2.2 Measurement Context: “We assume that ~~that~~ all”
* Sec 3.1 Overview: “each frame is comprises one” -> each frame comprises of one
* Sec 3.2 Video traffic identification: “We filter packets with ~~with~~”
* Sec 3.3 Estimation of frame boundaries. Dangling sentence: “approach is not used to fragment packets”
* Sec 3.3 Estimation of frame boundaries (second para): “Figure 1 shows the CDF of size difference in consecutive intra-frame…”. I think this Figure reference is incorrect, Figure 1 shows the CDFs of size of different traffic types (audio, video, video-retransmissions). Is this graph completely missing from the paper or is Figure 6 the correct reference?
* Sec 4.1 “the remaining traces have substantial variances in speed”. Though it is called a speed test, It is likely better to use throughput as the metric to describe the characteristics rather than speed.
* Sec 4.1 Matching ground truth with estimates. Figure ?? shows the distribution of the… Where is this figure in the paper?
* Sec 4.2 Frame Rate. The reference to Figure 6a is wrong, I think the correct reference is likely Figure 5a.
* Sec 4.2 “As a result, these frames would be combined with previous frames”. What does this sentence mean? Are they combined in terms of transmitted packets? Or are they not detected as separate frames by your approach?



Review #171C
===========================================================================

Overall merit
-------------
2. Weak reject

Reviewer expertise
------------------
4. Expert

Paper summary
-------------
The goal of this (short) paper is to infer the QoE of a real-time video applications such as Teams and Meet. Measuring QoE for video streaming is a solved problem, bit this paper claims that QoE metric is not easy to infer for real time video applications (although I wasnt completely clear about why inferring QoE is hard in real time)

Given a stream of video traffic data, this paper first identifies the RTP traffic and then separates out the video and audio components based on the size of the packet. For the video packets, the paper estimate the frame boundaries. It turns out that existing real time video applications divide frames into equal sized packets. So the paper use inter-packet size difference to infer frames. Based on this information, the paper computes the bitrate, frame rate, and jitter (jitter is estimated as the standard deviation between the frame end times)

Reasons to Accept
-----------------
The fraction of real time video traffic in the Internet is increasing, studying the performance of these applications is valuable. The paper performs real measurements over Teams and Meet.

Reasons to Reject
-----------------
I am not sure why computing QoE for real time video is difficult. The paper says that streaming videos use large buffers and can recover from small losses while real time videos cannot. Sure, but what does that have to do with estimating QoE?

Comments for author
-------------------
Beyond that, this paper does the most obvious thing to estimate QoE, which is to find out the bitrate and time taken to download each frame and use these estimates to measure QoE. The interesting thing here is to figure out the frame boundaries. But I didnt quite understand how the frame boundaries are estimated. If frames are segmented to equal sized packets, how can you figure out the frame boundaries? How is the inter-packet size difference used here?

In the end, the motivation was not clear, the contribution is rather small, and the one interesting part of the paper is not described well.



Review #171D
===========================================================================

Overall merit
-------------
1. Reject

Reviewer expertise
------------------
3. Knowledgeable

Paper summary
-------------
The paper uses passive network measurements to estimate VCA QoE metrics.

Reasons to Accept
-----------------
Understanding video QoE is important

Reasons to Reject
-----------------
Unfortunately, it is unclear how a network operator may apply the methodology as it does not work for sampled packets which is what network operators have available.
There seem to be a lot of invalidated assumptions, e.g., that most video re-transmissions are fixed in size at 304 bytes....

Comments for author
-------------------
Given the stated motivation of the paper that network operators may want to diagnose and react to QoE degradations, this reviewer is surprised by the assumptions that the paper makes, namely, that full traces of all packets are available to them. This is typically not the case. Most ISP do packet sampling, using IPFIX or NetFlow, which limits the available data.

Next, you are claiming that the focus is on application metrics such as frame rate and frame jitter. Yet, you seem to use network level metrics.

Moreover, this reviewer is confused why it is OK to disregard video resolution.

Now there are additional confusing assumptions: 
  - why is a video frame generated at the sender transmitted as soon as it has been encoded? Some systems are subject to network limitations or congestion and cannot do this. Others may not want to do this either ... Indeed, you also mention video re-transmissions later but do not discuss its impact
  - what happens if the PT types change?
  - what about screen sharing?
  - you seem to assume that all videos use RTP and that you have access to the timestamp info. Is that always the case? And how accurate are these timers?
  - do audio packets have to be smaller than video ones? Or is this an artifact of the specific encoding used?
  - what do video re-transmissions have to be fixed in size at 304 bytes? Isn't this another artifact that can change at any point in time?

How can you determine inter-packet size differences to identify frame boundaries if you have packet loss or only sampled data?

Why is it sufficient to consider two-person VCA calls? from where to where do you do these calls? How do you add network variability? How much data do you have available? of how many calls?

Moreover, there is a lot of related work out there that does do QoE estimation. 
How do you result compare to these if you apply them to your datasets?
For example: 
Wassermann, Sarah, et al. "Vicrypt to the rescue: Real-time, machine-learning-driven video-qoe monitoring for encrypted streaming traffic." IEEE Transactions on Network and Service Management 17.4 (2020): 2007-2023.



Review #171E
===========================================================================

Overall merit
-------------
1. Reject

Reviewer expertise
------------------
3. Knowledgeable

Paper summary
-------------
This paper derives quality influence factors (i.e., video bitrate, frame rate, and frame jitter) from unencrypted and encrypted RTP video streams.

Reasons to Accept
-----------------
+ Yet another paper that studies quality indicators of RTP video streams

Reasons to Reject
-----------------
- The paper doesn't go beyond extracting trivial-to-assess but relevant metrics (e.g., there is no attempt to derive a QoE model)
- Not well embedded into the existing QoE related work
- The robustness of the approach depends on having no capture loss, which is not realisitc in ISP-level deployments

Comments for author
-------------------
- The paper does not estimate QoE - it estimates QoE indicators. This is a very relevant difference since the output is not a perceptive score, but KPI's that can be mapped to a perceptive score by using a QoE model. Estimating KPI's that impact QoE is relevant and a classical approach. That is, there is nothing wrong, the paper should just avoid making the impressing that it infers a perceptive score (which is what QoE is all about). As such, the presentation should avoid the term "QoE metric" since there is no perceptive model in this work.

- The approach in this paper is very simplistic. The first approach (using RTP headers, Section 3.2) directly infers frame-level properties from RTP headers. This is direct extraction of header-level data and quite standard. The second approach (only using IP and UDP headers, Section 3.3) is quite trivial. This is a benefit and not necessarily a short coming. Yet, is demonstrates how easy it is to derive these metrics. The evaluation supports this: yes, it works well to parse this information from header data (which is not surprising and expected).

- The robustness of the approach depends on having no capture loss, which is not realistic. Even when excluding sampling (which is needed at high speed links), the paper makes the assumption that no packet is lost during the capturing process; otherwise the relative metrics (e.g., inter-packet size) cannot be derived.

- The paper is not very ambitious. ITU standards such as P.1202 attempt to derive perceptive scores from an RTP bitstream. The ITU standard describes a complex model to derive an QoE score from a UDP-based video stream, in which the bitstream model is one aspect. P.1202 hasn't been designed with video conferencing in mind and there might be differences that are worth pointing out. Yet, it has to be noted that the bitstream model described in this paper is a very simplistic one and less ambitious than those ITU standards (there is also another one for HTTP Adaptive Streaming: P.1204) that aim to derive a perceptive score (that is a true QoE model).
