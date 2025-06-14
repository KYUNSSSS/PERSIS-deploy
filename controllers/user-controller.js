// Import
const db = require("../models");
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const Validator = require('validator');

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


//Any Login
exports.login = async (req, res) => {
    if (Validator.isEmpty(req.body.username)) {

        return returnError(req, 400, { username: "USERNAMEISREQUIRED" }, res)
    }
    if (Validator.isEmpty(req.body.password)) {
        return returnError(req, 400, { password: "PASSWORDISREQUIRED" }, res)

    }
    if (req.headers["api-key"] != process.env.API_KEY) return returnError(req, 500, "INVALIDKEY", res);

    const username = req.body.username;
    const password = req.body.password;
    await user.findOne({
        where: {
            username: username
        }, raw: true
    }).then(async data => {
        if (data) {

            bcrypt.compare(password, data.password).then(async isMatch => {
                if (isMatch) {
                    const t = await connection.sequelize.transaction();

                    let payload = { id: data.id, username: data.username, iat: Date.now() };

                    jwt.sign(
                        payload,
                        "secret",
                        async (err, token) => {
                            // Write / Update into Login Table - SSO Purpose
                            await usrlgnpf.findOne({
                                where: {
                                    username: data.username
                                }, raw: true
                            }).then(async lgnpf => {
                                if (lgnpf) {
                                    let options = {
                                        pslgnsts: true,
                                        pslgidat: new Date(),
                                        pslgntkn: "Bearer " + token
                                    };
                                    if (lgnpf.pslgnsts) {
                                        options.pslgidat = new Date();
                                    }
                                    await usrlgnpf.update(options, {
                                        transaction: t,
                                        where: {
                                            id: lgnpf.id
                                        }
                                    }).catch(async err => {
                                        console.log(err);
                                        await t.rollback();
                                        return returnError(req, 500, "UNEXPECTEDERROR", res);
                                    });;
                                } else {
                                    await usrlgnpf.create({
                                        username: data.username,
                                        pslgnsts: true,
                                        pslgidat: new Date(),
                                        pslgntkn: "Bearer " + token
                                    }
                                    ).catch(async err => {
                                        console.log(err);
                                        await t.rollback();
                                        return returnError(req, 500, "UNEXPECTEDERROR", res);
                                    });
                                }
                            });



                            t.commit();
                            return returnSuccess(200, { token: "Bearer " + token }, res);
                        });

                } else return returnError(req, 400, { password: "INCORRECTPASS" }, res);

            });
        }
        else return returnError(req, 400, { username: "USERNOTFOUND" }, res);
    }).catch(err => {
        console.log(err);
        return returnError(req, 500, "UNEXPECTEDERROR", res);
    });
}


exports.change_password = async (req, res) => {
    const oldpassword = req.body.oldpassword
    const newpassword = req.body.newpassword

    req.body.newpassword = req.body.newpassword ? req.body.newpassword : "";
    req.body.oldpassword = req.body.oldpassword ? req.body.oldpassword : "";
    req.body.conpassword = req.body.conpassword ? req.body.conpassword : "";
    //Validation
    if (Validator.isEmpty(req.body.newpassword)) {
        // errors.newpassword = "FIELDISREQUIRED";
        return returnError(req, 400, { newpassword: 'FIELDISREQUIRED' }, res);
    } else {
        if (req.body.newpassword.length > 100) return returnError(req, 400, { newpassword: 'INVALIDVALUELENGTH&100' }, res);
        if (req.body.oldpassword == req.body.newpassword) {
            return returnError(req, 400, { newpassword: 'PASSWORDCANNOTSAME' }, res);
        }
    }

    if (Validator.isEmpty(req.body.conpassword)) {
        return returnError(req, 400, { conpassword: 'FIELDISREQUIRED' }, res);

    } else if (req.body.newpassword != req.body.conpassword) {
        return returnError(req, 400, { conpassword: 'PASSWORDMNOTMATCH' }, res);
    }

    user.findOne({
        where: {
            username: req.user.username
        }, raw: true, attributes: { exclude: ['createdAt', 'updatedAt', 'crtuser', 'mntuser'] }
    }).then(async user1 => {
        if (!user1) return returnError(req, 400, { username: 'USERNOTFOUND' }, res);

        await bcrypt.compare(oldpassword, user1.password).then(async isMatch => {
            if (isMatch) {
                // Change Password
                bcrypt.genSalt(10, (err, salt) => {
                    if (err) throw err;
                    bcrypt.hash(newpassword, salt, (err, hash) => {
                        if (err) throw err;
                        let newPassword = hash;
                        user.update({
                            password: newPassword,

                        }, {
                            where: {
                                id: user1.id
                            }
                        }).then(async () => {
                            return returnSuccessMessage(req, 200, "PASSWORDRESET", res);
                        }).catch(err => {
                            console.log(err);
                            return returnError(req, 500, "UNEXPECTEDERROR", res);
                        });
                    });
                });
            } else return returnError(req, 400, { oldpassword: "INVALIDPASSWORD" }, res);
        })
    }).catch(err => {
        console.log(err);
        return returnError(req, 500, "UNEXPECTEDERROR", res);
    });
}

exports.create = async (req, res) => {
    //Validation
    if (Validator.isEmpty(req.body.username)) {
        return returnError(req, 400, { username: "FIELDISREQUIRED" }, res);
    } else {
        if (req.body.username.length > 255)
            return returnError(req, 400, { username: "INVALIDVALUELENGTH&255" }, res);

    }

    if (Validator.isEmpty(req.body.name)) {
        return returnError(req, 400, { name: "FIELDISREQUIRED" }, res);
    } else {
        if (req.body.name.length > 255)
            return returnError(req, 400, { name: "INVALIDVALUELENGTH&255" }, res);

    }
    if (Validator.isEmpty(req.body.age)) {
        return returnError(req, 400, { age: "FIELDISREQUIRED" }, res);
    } else {
        if (req.body.age > 150) return returnError(req, 400, { age: "INVALIDVALUELENGTH&150" }, res);

    }
    if (Validator.isEmpty(req.body.gender)) {
        return returnError(req, 400, { gender: "FIELDISREQUIRED" }, res);
    } else {
        if (req.body.gender.length > 1)
            return returnError(req, 400, { gender: "INVALIDVALUELENGTH&1" }, res);
    }
    if (req.body.gender != "M" && req.body.gender != "F") {
        return returnError(req, 400, { gender: "INVALIDDATAVALUE" }, res);


    }


    // Check Duplicate
    let user1 = await user.findOne({
        where: {
            username: req.body.username
        }, raw: true
    });
    if (user1) return returnError(req, 400, { username: 'USERALREADYEXIST' }, res);


    const new_user = {
        username: req.body.username,
        name: req.body.name,
        age: req.body.age,
        gender: req.body.gender,
        recordPath: req.body.username + "-user.csv"

    }

    bcrypt.genSalt(10, (err, salt) => {
        if (err) throw err;
        bcrypt.hash(req.body.password, salt, (err, hash) => {
            if (err) throw err;
            new_user.password = hash;
            user.create(new_user).then(() => {
                return returnSuccessMessage(req, 200, "RECORDCREATED", res);
            }).catch(err => {
                console.log(err);
                return returnError(req, 500, "UNEXPECTEDERROR", res);
            });
        })
    });
}



exports.detail = (req, res) => {
    const usrnm = req.query.id ? req.query.id : '';
    if (usrnm == '') return returnError(req, 500, "RECORDIDISREQUIRED", res);
    user.findOne({
        where: {
            username: usrnm
        }, raw: true, attributes: [['id', 'username'], 'username', 'name', 'gender', 'age']
    }).then(async usrunme => {
        if (usrunme) {

            return returnSuccess(200, usrunme, res);
        } else return returnError(req, 500, "NORECORDFOUND", res);
    }).catch(err => {
        console.log(err);
        return returnError(req, 500, "UNEXPECTEDERROR", res);
    });
}

exports.update = async (req, res) => {
    // if (req.user.psusrtyp != 'ADM')
    //     return returnError(req, 500, "INVALIDAUTHORITY", res);

    //Validation
    req.body.username = req.body.username ? req.body.username : "";
    req.body.name = req.body.name ? req.body.name : "";
    req.body.age = req.body.age ? req.body.age : "";
    req.body.gender = req.body.username ? req.body.gender : "";
    if (Validator.isEmpty(req.body.username)) {
        return returnError(req, 400, { username: "FIELDISREQUIRED" }, res);
    }

    if (Validator.isEmpty(req.body.id)) {
        return returnError(req, 400, { id: "FIELDISREQUIRED" }, res);
    }
    // else {
    //     if (req.body.username.length > 255) errors.username = 'INVALIDVALUELENGTH&255';
    // }

    if (Validator.isEmpty(req.body.name)) {
        return returnError(req, 400, { name: "FIELDISREQUIRED" }, res);
    } else {
        if (req.body.name.length > 255)
            return returnError(req, 400, { name: "INVALIDVALUELENGTH&255" }, res);

    }
    if (Validator.isEmpty(req.body.age)) {
        return returnError(req, 400, { age: "FIELDISREQUIRED" }, res);
    } else {
        if (req.body.age > 150) return returnError(req, 400, { age: "INVALIDVALUELENGTH&150" }, res);

    }
    if (Validator.isEmpty(req.body.gender)) {
        return returnError(req, 400, { gender: "FIELDISREQUIRED" }, res);
    } else {
        if (req.body.gender.length > 1)
            return returnError(req, 400, { gender: "INVALIDVALUELENGTH&1" }, res);
    }
    if (req.body.gender != "M" && req.body.gender != "F") {
        return returnError(req, 400, { gender: "INVALIDDATAVALUE" }, res);


    }



    await user.findOne({
        where: {
            username: req.body.id
        },
        raw: true, attributes: { exclude: ['createdAt', 'updatedAt', 'crtuser', 'mntuser'] }
    }).then(async user1 => {
        if (!user1) return returnError(req, 400, { username: "USERNOTFOUND" }, res);

        let obj = req.body;



        const new_user = {
            username: obj.username,
            name: obj.name,
            age: obj.age,
            gender: obj.gender,
        }

        await user.update(new_user,
            {
                where: {
                    id: user1.id
                }
            }).then(() => {

                return returnSuccessMessage(req, 200, "USERUPDATED", res);
            }).catch(err => {
                console.log(err);
                return returnError(req, 500, "UNEXPECTEDERROR", res);
            });
    }).catch(err => {
        console.log(err);
        return returnError(req, 500, "UNEXPECTEDERROR", res);
    });
}

exports.logout = async (req, res) => {
    try {
        const t = await connection.sequelize.transaction();
        
        await usrlgnpf.update({
            pslgnsts: false,
            pslgidat: new Date(),
            pslgntkn: null
        }, {
            where: {
                username: req.user.username
            },
            transaction: t
        });

        await t.commit();
        return returnSuccessMessage(req, 200, "LOGOUTSUCCESSFUL", res);
    } catch (err) {
        console.log(err);
        if (t) await t.rollback();
        return returnError(req, 500, "UNEXPECTEDERROR", res);
    }
};

const isEmpty = (value) =>
    value === undefined ||
    value === null ||
    (typeof value === 'object' && Object.keys(value).length === 0) ||
    (typeof value === 'string' && value.trim().length === 0)
