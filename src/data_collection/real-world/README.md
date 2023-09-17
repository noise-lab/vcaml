# Automating Video Conferencing Calls

This tool automates UChicago Zoom calls and all google meet calls. Users can 
capture network traffic during calls using command line options.

## Dependencies

This tool assumes [geckodriver](https://github.com/mozilla/geckodriver/releases) is in PATH. 

Requires [selenium](https://pypi.org/project/selenium/). 

### .netrc

`vcqoe` accesses login information that is stored in a .netrc file. Add the 
following to your .netrc, choosing either zoom or meet and filling in
your username and password.

`machine [zoom/meet]`
`login [username]`
`password [password]`


## Usage

Use `python3 vcqoe -h` to view required and optional command line arguments.
You must be in the `src` directory to run `vcqoe`.

### Zoom

To launch a zoom call, use the following command:

`python3 vcqoe [website] -fp [path/to/your/firefox/profile] -id [url_of_zoom_meeting]`

There are also the following network capture options:
- `-i` will run vcqoe as a headless browser.
- `-c` will capture network traffic during the call.
- `-f [filter]` will filter the captured network traffic based on 
	the given libpcap `[filter]`.
- `-n [num_trials]` sets the number of calls to make. Default is 1.
- `-t [call_length]` sets the length of each call in secs. Timer 
	begins once zoom/meet call has begun. Default is 60.

Use `python3 vcqoe -h` to view required and optional command line arguments. Tool must receive website, username, password, and 
zoom meeting url. Any meeting with a password must have password included in the url. 

Must be in `src` directory before running commands. 

