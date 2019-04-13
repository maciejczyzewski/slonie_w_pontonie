import webbrowser

f = open('helloworld.html', 'w')

message = """
<!doctype html>
<html lang=”en”>
<head>
<meta charset="utf-8">
<meta http-equiv="x-ua-compatible" content="ie=edge">
<meta name="viewport" content="width=device-width, initial-scale=1">
<!-- Name of your awesome camera app -->
<title>Camera App</title>
<!-- Link to your main style sheet-->
<link rel="stylesheet" href="style.css">
</head>
<body>
<!-- Reference to your JavaScript file -->
<script src="app.js"></script>
</body>
</html>
"""

f.write(message)
f.close()

webbrowser.open_new_tab('helloworld.html')
