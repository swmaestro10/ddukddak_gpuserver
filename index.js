let express = require('express');

let app = express();

app.get('/', function(request, response) {

    response.send('server online');

});

app.listen(80, () => console.log('DDUK-DDUAK-Learning GPU Server running on port 80!'));
