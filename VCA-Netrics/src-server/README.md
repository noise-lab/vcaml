Code for running the other end point that performs the following functions 
- Runs a web server 
- The client queries the server whether it can run a test
- If the server is busy, it will reply no and the client will try after certain time
- If the server is available, it randomly selects between a meet or teams call and provides a url to the client
- At the same time, it will start a video call for the corresponding vca. In addition, it will also pipe a video and start network capture
- Finally, it will admit the other end point
