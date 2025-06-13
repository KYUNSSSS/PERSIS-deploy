const express = require("express");
// const { spawn } = require("child_process");
const db = require("./models")
const passport = require('passport');
require('./config/passport')(passport);
// const _ = require('lodash');
const app = express();

app.use(express.json());
app.use(passport.initialize());
//controller execution
// const user = db.user;
// async function runPythonProcess(data, res) {
//     try {

//         const userFind = await user.findOne({
//             where: {
//                 username: data.username
//             },
//             raw: true,
//             attributes: ['username', 'name']
//         });

//         // Handle user not found scenario
//         if (!userFind || _.isEmpty(userFind.name)) {
//             return res.status(400).json({ response: "usernotfound" });
//         }


//         const pythonProcess = spawn("python", ["./QL/app.py"]);

//         pythonProcess.stdin.write(JSON.stringify(data));
//         pythonProcess.stdin.end();

//         let responseData = "";

//         pythonProcess.stdout.on("data", (chunk) => {
//             console.log("Raw Python Response:", chunk.toString());
//             responseData += chunk.toString();
//         });


//         pythonProcess.stderr.on("data", (data) => {
//             console.error("Python Error:", data.toString());
//         });



//         pythonProcess.on("close", (code) => {
//             if (!responseData.trim()) {
//                 console.error("No response received from Python script.");
//                 return res.status(500).json({ response_type: 3, response: "UnexpectedError" });
//             }

//             try {
//                 const jsonResponse = JSON.parse(responseData.trim()); // Ensure complete JSON
//                 return res.json(jsonResponse);
//             } catch (error) {
//                 console.error("JSON Parsing Error:", error);
//                 return res.status(500).json({ response_type: 3, response: "UnexpectedError" });
//             }
//         });


//     } catch (error) {
//         console.error("Error in processing:", error);
//         return res.status(500).json({ response: "UnexpectedError" });
//     }
// }


// app.post("/invoke", (req, res) => {
//     runPythonProcess(req.body, res);
// });


const userroutes = require("./routes/user");
app.use('/api/user', userroutes);

const invoke = require("./routes/invoke");
app.use('/api/invoke', invoke);



app.use(function (req, res, next) {
    res.status(404).send("APINOTFOUND, This is an default API");
});



console.log("Server started=====>");
// app.listen(3000, () => {
//     console.log("Server running on port 3000");
// });
db.sequelize.sync() // Ensure database sync
    .then(() => {
        console.log("Database synced successfully");
        app.listen(3000, () => {
            console.log("==========================================");
            console.log("Server running on http://localhost:3000/");
            console.log("==========================================");
        });
    })
    .catch((err) => {
        console.error("Error syncing database:", err);
    });
