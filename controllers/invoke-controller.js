// Import
const db = require("../models");
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const Validator = require('validator');
const { spawn } = require("child_process");
const _ = require('lodash');

//Table import
const usrlgnpf = db.usrlgnpf;
const user = db.user;


//Common Function
// Common Function
const Op = db.Sequelize.Op;
const returnError = require('../common/error');
const returnSuccess = require('../common/success');
const returnSuccessMessage = require('../common/successMessage');
const connection = require("../common/db");

// Input Validation


exports.runPythonProcess = async (req, res) => {
    try {
        const username = req.user.username;
        if (!req.user.username) return returnError(req, 500, { username: "UNEXPECTEDERROR" }, res);
        if (!req.body.username) req.body.username = username;

        const userFind = await user.findOne({
            where: {
                username: username
            },
            raw: true,
            attributes: ['username', 'name']
        });

        // Handle user not found scenario
        if (!userFind || _.isEmpty(userFind.name)) {
            return returnError(req, 500, { username: "USERNOTFOUND" }, res);
            // return res.status(400).json({ response: "USERNOTFOUND" });
        }


        const pythonProcess = spawn("python", ["./QL/app.py"]);

        pythonProcess.stdin.write(JSON.stringify(req.body));
        pythonProcess.stdin.end();

        let responseData = "";

        pythonProcess.stdout.on("data", (chunk) => {
            console.log("Raw Python Response:", chunk.toString());
            responseData += chunk.toString();
        });


        pythonProcess.stderr.on("data", (data) => {
            console.error("Python Error:", data.toString());
        });



        pythonProcess.on("close", (code) => {
            if (!responseData.trim()) {
                console.error("No response received from Python script.");
                return res.status(500).json({ response_type: 3, response: "UnexpectedError" });
            }

            try {
                const jsonResponse = JSON.parse(responseData.trim()); // Ensure complete JSON
                return res.json(jsonResponse);
            } catch (error) {
                console.error("JSON Parsing Error:", error);
                return res.status(500).json({ response_type: 3, response: "UnexpectedError" });
            }
        });


    } catch (error) {
        console.error("Error in processing:", error);
        return returnError(req, 500, "UNEXPECTEDERROR", res);
    }
}