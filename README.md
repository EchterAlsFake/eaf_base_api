# EAF Base API

# What is this?
When using one of my Porn site APIs, you probably came across this package and wondered what it actually does, so here's
a detailed answer. 

A lot of Porn sites use very similar methods for m3u8 (HLS) parsing and other things. I also wanted to implement proxy
support, and there was a lot of code that I would have rewritten in every API again and again. That's why I made this API
package. The `BaseCore` class does all the necessary stuff like m3u8 parsing, a great caching system, network request
fetching with retry attempts and proxy support.

# Using Proxies / Caching
There's a small documentation for this project.
<br>Please have a look at it: https://github.com/EchterAlsFake/API_Docs/blob/master/Porn_APIs/eaf_base_api.md


# Can I use this for myself?
Yes, you can, but I may change stuff here and there from time to time, and it would maybe break your project.
I would not recommend you to install and use it as a package, but just copy the code you need.

I can recommend everyone the download functions for HLS streaming since, for example, the threaded preset is very well 
optimized. If you just use mine, you need to consume less caffeine and brain cells to make such a function :)


# License
Licensed under The [LGPLv3](https://www.gnu.org/licenses/lgpl-3.0.en.html) license.
<br>Copyright (C) 2024-2025 Johannes Habel
