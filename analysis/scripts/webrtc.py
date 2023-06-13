import json, csv 
import glob
import pandas as pd 
from dateutil.parser import parse
import math 
import os
import copy
import numpy as np
import argparse
import pathlib 


#144 empty 
#27 nothing

#Globals
VCA = "meet"
NC = "static"
ssrc_trackId_mapping = dict()

#Only for newly collected dynamic dataset 
#Record 
def mapSSRCtoTrackID(mapping,webRTC_stats):
    #Fill key of map dict with trackID stat title instances 
    trackIds = [tid for tid in webRTC_stats if (("trackId" in tid) and ("RTCInboundRTPVideoStream" in tid))]
    for x in trackIds:
        mapping[x.split("_")[1].split("-")[0]] = json.loads(webRTC_stats[x]['values'])[0]

    return mapping 


#Given SSRC 
#returns the number used in webrtc stats, this number is part of the tile for RTCMediaStreamTrack_receiver
def lookUpSSRC(SSRC,webRTC_stats):
    #check every track identifier 
    for stat in webRTC_stats:
        if "static" in NC:
            if (("RTCMediaStreamTrack_receiver_" in stat) and ("trackIdentifier" in stat)):
                #continue #take out when want freeze count 
                #if dataset has static network conditions 
                new_ssrc = json.loads(webRTC_stats[stat]['values'])[0]
                if "team" in VCA:
                    new_ssrc = new_ssrc.split("-")[1]
                    
        #If dataset dynamic
        else:
            if ("RTCMediaStreamTrack_receiver_" in stat):
                ID = stat.split("-")[0]
                try:
                    new_ssrc = list(ssrc_trackId_mapping.keys())[list(ssrc_trackId_mapping.values()).index(ID)]
                except:
                    continue

        try:
            if int(SSRC) == int(new_ssrc):
                trackIdentifier = stat.split("-")[0].split("_")[-1]
                return trackIdentifier
        except UnboundLocalError:
                #If "new_ssrc" referenced before assignment
                continue 
    
    return None 
            

#Only keep statistic features in which the ssrc has all the features we want 
def removeBadStats(all_stats, dict_output, total_stats,webRTC_stats):

    for stat in all_stats:
        if isinstance(stat,list):
            all_stats.remove(stat)
    
    refined_stats = copy.deepcopy(all_stats)
    for ssrc in dict_output.keys():
        count = len([stat for stat in all_stats if ((ssrc in stat) or (lookUpSSRC(ssrc,webRTC_stats) == stat.split("-")[0].split("_")[-1]) or ())])  #(int(ssrc) == int(json.loads(webRTC_stats[stat]['values'])[0])) 
        dict_output[ssrc] = count 


    tmp_dict_output = copy.deepcopy(dict_output)
    #Remove stats associated with the ssrc that doesnt have all stats
    for key, value in tmp_dict_output.items():
        bad_ssrc = str(-1)
        #if this ssrc doesnt have all stats, record all stats that's not associated with the bad ssrc 
        #added if(value != (total_stats+1)) to below for newer collected dataset since 
        #lastPacketReceivedTimes has 2 fields: lastPacketReceivedTimes & [lastPacketReceivedTimes] 

        if (value != total_stats) and ("static" in NC):
            bad_ssrc = key
        elif ((value != (total_stats+1)) or (value != total_stats)) and ("dynamic" in NC):
            bad_ssrc = key
        
        
        for stat in all_stats: 
            if ("RTCMediaStreamTrack_receiver_" in stat): 
                if "static" in NC:
                    try:
                        intID = lookUpSSRC(bad_ssrc,webRTC_stats)
                        if str(intID) in stat:
                            refined_stats.remove(stat) 
                        else:
                            continue
                    except:
                        continue 
                else:
                    ID = stat.split("-")[0]
                    try:
                        new_ssrc = list(ssrc_trackId_mapping.keys())[list(ssrc_trackId_mapping.values()).index(ID)]
                        if (new_ssrc==bad_ssrc):
                            refined_stats.remove(stat) 
                    except:
                        continue 
    
            else:
                if (bad_ssrc in stat): 
                    refined_stats.remove(stat)

        try:
            dict_output.pop(bad_ssrc)
        except:
            pass 

    for stat in refined_stats:
        if isinstance(stat,list):
            refined_stats.remove(stat)

    return dict_output, refined_stats


def getActiveStatsWebRTC(webRTC_stats,dict_stats,manual_stats):
    all_stats = set()
    ssrc_list = list()

    #Get list of all ssrc
    for key, value in dict_stats.items():
        if key != "RTCMediaStreamTrack_receiver":
            ssrc_list = ssrc_list + list(ssrc.split("_")[1].split("-")[0] for ssrc in webRTC_stats if ((key in ssrc) and ("ssrc" in ssrc)))
            
        else:
            #ssrc_list = ssrc_list + list(ssrc.split("_")[2].split("-")[0] for ssrc in webRTC_stats if ((key in ssrc)))
            continue
    #Make dict to hold count of statistical features per ssrc 
    ssrc_featureCount = dict.fromkeys(ssrc_list,0)
    #Get list of all webRTC statistic names given the desired statistics

    #Holds the total number of distinct statistics 
    total_stats = 0
    for key, value in dict_stats.items():
        for v in value:
            if v in manual_stats:
                continue
            else:
                total_stats += 1 

            for stat in webRTC_stats:
                if ((key in stat) and (v in stat)):
                    if ("RTCMediaStreamTrack_receiver_" in stat):
                        if "static" in NC:
                            for ssrc in ssrc_list:
                                try:
                                    intID = lookUpSSRC(ssrc,webRTC_stats)
                                    if str(intID) in stat:
                                        all_stats.add(stat)
                                    else:
                                        continue
                                except:
                                    continue
                        else:
                            ID = stat.split("-")[0]
                            for ssrc in ssrc_list:
                                try:
                                    new_ssrc = list(ssrc_trackId_mapping.keys())[list(ssrc_trackId_mapping.values()).index(ID)]
                                    if (new_ssrc==ssrc):
                                        all_stats.add(stat)
                                except:
                                    continue 
                    else:
                        all_stats.add(stat)
                
    all_stats = list(all_stats)
    
    ssrc_featureCount, all_stats = removeBadStats(all_stats,ssrc_featureCount,total_stats,webRTC_stats)


    if all_stats is None:
        return ssrc_featureCount, all_stats

    tmp_all_stats = copy.deepcopy(all_stats)

    #Remove statistical features for each SSRC that its len(values)! = that SSRC's lastPacketReceivedTimestamp's len(values)
    #Get all distinct lastPacketReceivedTimestamp, hence also distinct SSRCs
    for ssrc, c in ssrc_featureCount.items():
        for stat in tmp_all_stats:
            if (str(ssrc) in stat) or (str(lookUpSSRC(ssrc,webRTC_stats)) in stat):
                try:
                    if (len(json.loads(webRTC_stats[stat]['values'])) != len(json.loads(webRTC_stats["RTCInboundRTPVideoStream_"+str(ssrc)+"-lastPacketReceivedTimestamp"]['values']))):
                        diff = len(json.loads(webRTC_stats[stat]['values']))-len(json.loads(webRTC_stats["RTCInboundRTPVideoStream_"+str(ssrc)+"-lastPacketReceivedTimestamp"]['values']))
    
                        if (diff > 5) or (diff < -5): 
                            all_stats.remove(stat)
                    
                except:
                    all_stats.remove(stat)

    ssrc_featureCount, all_stats = removeBadStats(all_stats,ssrc_featureCount,total_stats,webRTC_stats)
    return ssrc_featureCount, all_stats


def extractWebRTC(webRTC_stats,stats):
    manual_stats = ["startTime", "endTime"]

    ssrc_list, all_webrtc_stats = getActiveStatsWebRTC(webRTC_stats,stats,manual_stats)

    if bool(ssrc_list):
        #Make output df 
        stats_iter = iter(stats)
        output = {k : [] for k in stats[next(stats_iter)]}
        output.update({k : [] for k in stats[next(stats_iter)]}) #since 2 keys in dict

        ssrc_stats = {}
        len_ssrc_stats = {}
        for ssrc in ssrc_list.keys():
            for stat_name in all_webrtc_stats:
                if (ssrc in stat_name) or ((lookUpSSRC(ssrc,webRTC_stats) ==  stat_name.split("-")[0].split("_")[-1])):
                    ssrc_stats[stat_name] = json.loads(webRTC_stats[stat_name]['values'])
                    len_ssrc_stats[stat_name] = len(json.loads(webRTC_stats[stat_name]['values']))

            #print(len_ssrc_stats)

            #Check which stat has the least length 
            min_stat = min(len_ssrc_stats.values())
            target_stat = [key for key in ssrc_stats if len(ssrc_stats[key]) == min_stat][0]
            input_stats = [key for key in ssrc_stats if len(ssrc_stats[key]) != min_stat]

            for input_stat in input_stats:
                values = json.loads(webRTC_stats[input_stat]['values'])
                for i in range(len(values)-min_stat):
                    values.pop(0)
                ssrc_stats[input_stat] = values 
            
            
            #Fill values of stats (without manual stats)
            #If unequal features length for the same ssrc, skip to next ssrc
            some_length = len(list(ssrc_stats.items())[0][1])
            if all(some_length == l for l in [len(value) for key,value in ssrc_stats.items()]):
                for key, value in output.items():
                    if key in manual_stats:
                        continue
                    for k,v in ssrc_stats.items():
                        if ("-"+key) in k:
                            output[key].extend(ssrc_stats[k])

            #Fill values for manual stats
            for man_stat in manual_stats:
                if (man_stat == "startTime") or (man_stat == "endTime"):                    
                    values_list = [int(parse(webRTC_stats["RTCInboundRTPVideoStream_{}-lastPacketReceivedTimestamp".format(ssrc)][man_stat]).timestamp()) for i in range(len(ssrc_stats["RTCInboundRTPVideoStream_{}-lastPacketReceivedTimestamp".format(ssrc)]))]
                    output[man_stat].extend(values_list)

            #delete ssrc_stats 
            del ssrc_stats 
            del len_ssrc_stats

        if output["ssrc"]:
            #Dict -> PD 
            output = pd.DataFrame.from_dict(output)
        else:
            output = pd.Series([np.NaN])
    else:
        output = pd.Series([np.NaN])

    #For older webrtc: lastPacketReceivedTimestamp's values are 6 digits 
    #and if comparing first value to startTime's last 6 digits, it's always off by 62434 seconds where startTime is less
    #For newer webrtc, lastPacketReceivedTimestamp is in milliseconds 
    return output 

def get_webrtc(filename):
    webrtc = json.load(open(filename))
    try: 
        unknown_key = list(webrtc["PeerConnections"].keys())[0]
    except IndexError:
        #If json file empty 
        continue 

    webRTC_stats = webrtc["PeerConnections"][unknown_key]["stats"]


    wantedStats = {"RTCInboundRTPVideoStream" : ["ssrc", "startTime", "endTime", "lastPacketReceivedTimestamp", \
                    "framesPerSecond","[bytesReceived_in_bits/s]", "[codec]", "packetsLost", "framesDropped"], \
                    "RTCMediaStreamTrack_receiver" : ["trackIdentifier", "freezeCount*","totalFreezesDuration*"]}


    output = extractWebRTC(webRTC_stats,wantedStats)

    #Order columns
    #output = reorderColumns(output,isInbound, isQoE, isQoS) 

    if isinstance(output, pd.DataFrame):
        if "static" in NC:
            if "team" in VCA:
                #print(filename)
                downlink_bandwidth = filename.split("/")[-1].split("-")[-2] #Mbps
                # print(downlink_bandwidth)
                uplink_bandwidth = filename.split("/")[-1].split("-")[-1].split(".json")[0] #Mbps
            else:

                #print(filename)
                downlink_bandwidth = filename.split("/")[-1].split("-")[-4] #Mbps
                # print(downlink_bandwidth)
                uplink_bandwidth = filename.split("/")[-1].split("-")[-3] #Mbps
                # print(uplink_bandwidth)
                latency_induced = filename.split("/")[-1].split("-")[-2] #ms
                # print(latency_induced)
                packetLoss_induced = filename.split("/")[-1].split("-")[-1].split(".json")[0] #%
                # print(packetLoss_induced)

                output["latencyInduced"] = latency_induced
                output["packetLossInduced"] = packetLoss_induced

            #Add network condition fields 
            #downlink_bandwidth, uplink_bandwidth, latency, packet loss (%) 
            output["downlinkBand"] = downlink_bandwidth
            output["uplinkBand"] = uplink_bandwidth

        #Pandas dataframe -> CSV 
        target_filename = str(target_dir) + "/" + filename.split("/")[-1].split(".json")[0] + ".csv"
        output.to_csv(target_filename,header=True,index=False)
        print("Saved to -> " + target_filename)

    ssrc_trackId_mapping.clear()