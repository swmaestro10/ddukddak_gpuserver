let express = require('express');
let bodyParser = require('body-parser');
let child_process = require('child_process');
let fs = require('fs');

let app = express();
app.use(bodyParser.text());

app.get('/code', function(request, response) {

    let code = request.body;

    console.log(code);

    let date = new Date();
    let timestamp =
        date.getFullYear().toString() +
        ((date.getMonth() + 1) < 10 ? ('0' + (date.getMonth() + 1)) : (date.getMonth() + 1)).toString() +
        (date.getDate() < 10 ? ('0' + date.getDate()) : date.getDate()).toString() +
        (date.getHours() < 10 ? ('0' + date.getHours()) : date.getHours()).toString() +
        (date.getMinutes() < 10 ? ('0' + date.getMinutes()) : date.getMinutes()).toString() +
        (date.getSeconds() < 10 ? ('0' + date.getSeconds()) : date.getSeconds()).toString();
    let location = __dirname + `/io/code/${timestamp}.py`;

    // save code to file
    fs.writeFile(location, code, function(err) {

        if(err) return console.log(err);

        // execute python
        child_process.exec(
            `python ${location}`,
            function (error, stdout, stderr) {

                if (error) {

                    response.writeHead(200);
                    response.end(error);

                } else {

                    response.writeHead(200);
                    response.end(stdout);

                }

            });

    });

});

app.listen(80, () => console.log('DDUK-DDUAK-Learning GPU Server running on port 80!'));
