from flask import Request, Response


class Middleware:
    """
    simple wsgi middleware that check user name and password
    """

    def __init__(self, app):
        self.app = app
        self.username = "admin"
        self.password = "thisisacomplicatedpassword"

    def __call__(self, environ, start_response):
        request = Request(environ)
        if request.authorization is not None:
            # get username and password
            username = request.authorization["username"]
            password = request.authorization["password"]

            # check username and password
            if username == self.username and self.password == password:
                return self.app(environ, start_response)

        # if username and password is wrong
        return Response("Authorization Failed", 401, {"WWW-Authenticate": "Basic"})(
            environ,
            start_response,
        )
