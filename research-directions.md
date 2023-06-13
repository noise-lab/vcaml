## Contributions

**Current Contributions**

This paper:
- develops a method to infer key VCA QoE
metrics at finer time granularities, with and without application headers;
- demonstrates that this method can
estimate FPS within two FPS for up to 87% of cases and video
bitrate within 50 kbps for up to 85% of cases using RTP header
fields; and estimate FPS within two FPS for up to 77% of cases
and video bitrate within 50 kbps for up to 81% of cases using
only IP/UDP headers.



### Possible Contributions

#### Algorithmic
- Design an approach that infers QoE using non-RTP headers at fine time
    granularity 
    -   Current approach assumes packetization is into equal-sized packets --
        packetization logic is dependent on the encoding standard. Another 
        insight is to use inter-arrival (IAT) time. But IAT is not network invariant.
    -   Explore machine learning aporaches with features based on the above
        insight (Packet size, and IAT)
- Evaluate the accuracy of the approach under diverse network conditions and
    application context
    -   Streaming Services -- Teams, Meet, and **Zoom**
    -   QoE metrics -- FPS, Video Frame Jitter, Video bitrate, **Frame-level
        latency, Video Resolution**
    -   Usage modality -- Two-person call, **>2 person call, Screen sharing,
        Speaker vs Gallery, Video blur, Video on/off, Audio on/off**
        - Consider which of these application modalities are important and how easy it is to scale?
        - Enumerate the complexity and value of incorporating each modality.  
    -   Network conditions -- (1), Controlled experiments with diverse network
        throughput and latency, **packet losses**; (2), Deployment in the wild
        including **UCSB campus and Chicago households**
        **Under controlled experiments** Explicity generate challenging cases:
        - Trigger FEC using uniform losses
        - Latency/Jitter
        - Zoom bandwidth probing mechanism
        - Throughput variation

#### VCA QoE Characterization
- Use developed techniques to characterize VCA QoE on campus. Both UCSB and
UChicago traffic. Interesting Questions:
- What is the performance of these applications in a real-world network?
  How often do QoE impairments occur?
- How do applications perform under congestions -- e.g., in a library?
- Can we augment network performance with wireless information? (@Arpit) and correlate how
    the wireless metrics impact application performance? 
- Comparative analysis of different VCAs. Can potentially uncover interesting
    design aspects. 

#### System Design
Goal: How to design a scalable VCA QoE inference system? 
Possible challenges: 
- Scale or optimize the computational cost.
- Augmentation with additional data for diagnostics. 
- Context dependent -- existing ISPs may have legacy monitoring systems. Can we
    develop approaches that work with existing systems? What about sampling
    techniques?

Thought exercise: Would existing approach work with UCSB traffic for real-time
monitoring? 






## Concrete Steps


#### Algorithmic: Validate if the existing technique works for Zoom (Taveesh, Tarun) 
**Motivation**
Zoom is quite popular VCA. Majority of UCSB traffic is Zoom. 
We need to validate if our approach works for Zoom.

**Challenge**
Need ground truth QoE data. Unlike Teams and Meet, Zoom uses Datachannels on
WebRTC.


**Approach**
- Use Zoom SDKs like the upcoming IMC paper
- Use an annotated video. This is a generic solution independent of the
    underlying VCA. This could also be a contribution. Q: Would it work for a RasPi?



- Start with an easy task (e.g., data processing pipelines)
- Taveesh can start analyzing the UCSB campus traffic 
 
#### Test on the data collected from UCSB / Netrics  (Nawel, Arpit)
- 

#### Test the technique under different usage modality (Taveesh, Tarun)
Different usage modalities include:
1. Number of users. Focus on the case when number of users increase from two to three
  With three or more users, most apps switch to centralized streaming.
    - Experiment: Consider a three-person call (A, B, C) with one participant (A) having a poor
        upload connectivity. What does QoE inference using C's passive network measurement 
        look like? Are there any differences in stream identification logic?
2. Screen sharing: Consider a two-person call with screen sharing on. How to
    detect packets corresponding to screen sharing?
3. Audio/Video ON and OFF: Can we detect whether audio/video is on or off from
    network data?
4. Different background: Are there differences when background is  Probably not that important.

In terms of effort, 3 and 4 should be quick. 1 would need a little bit of
programming to coordinate call across multiple devices. 


**Approach**
- [ ] Programmatically change usage modality
- [ ] Consider collecting data under normal usage where a RasPI randomly joins a regular meeting
