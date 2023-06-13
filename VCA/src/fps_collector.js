// ==UserScript==
// @name         New Userscript
// @namespace    http://tampermonkey.net/
// @version      0.1
// @description  try to take over the world!
// @author       You
// @match        https://meet.google.com/*-*-*
// @icon         https://www.google.com/s2/favicons?sz=64&domain=google.com
// @grant        none
// ==/UserScript==

var data = {}

console.log('VCA detected.');

var stats_collector = () => {
  var video_elements = document.getElementsByTagName('video');
  for (const element of video_elements){
    var timestamp = new Date().toISOString();
    var quality = element.getVideoPlaybackQuality();
    var measurement = {"total" : quality.totalVideoFrames, "dropped" : quality.droppedVideoFrames, "timestamp": timestamp};
    if (!(element.className in data)){
      data[element.className] = []
    }
    data[element.className].push(measurement);
  }
};

var exportToJson = (objectData) => {
    let filename = "export.json";
    let contentType = "application/json;charset=utf-8;";
    if (window.navigator && window.navigator.msSaveOrOpenBlob) {
      var blob = new Blob([decodeURIComponent(encodeURI(JSON.stringify(objectData)))], { type: contentType });
      navigator.msSaveOrOpenBlob(blob, filename);
    } else {
      var a = document.createElement('a');
      a.download = filename;
      a.href = 'data:' + contentType + ',' + encodeURIComponent(JSON.stringify(objectData));
      a.target = '_blank';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  }

var export_logs = (e) => {
    var evtobj = window.event? event : e;
    console.log('Key pressed');
    if (evtobj.keyCode == 89 && evtobj.ctrlKey && evtobj.shiftKey){
      console.log('ctrl+shift+y');
      console.log(data);
      exportToJson(data);
    }
}

(function() {
    'use strict';
    console.log("Extension active!");
    console.log("meet URL identified");
    document.onkeydown = export_logs;
    setInterval(() => stats_collector(), 1000);
})();
