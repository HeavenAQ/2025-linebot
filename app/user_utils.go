package app

import "github.com/HeavenAQ/nstc-linebot-2025/api/db"

func (app *App) createUser(userID string) *db.UserData {
	// Retrieve user's name from LINE
	app.Logger.Info.Println("Getting the user's name")
	username, err := app.LineBot.GetUserName(userID)
	if err != nil {
		app.Logger.Error.Println("Error getting new user's name:", err)
	}
	app.Logger.Info.Println("User name has been retrieved")

	// Create user's folders
	app.Logger.Info.Println("Creating the user's folders")
	userFolders, err := app.DriveClient.CreateUserFolders(userID, username)
	if err != nil {
		app.Logger.Error.Println("Error creating new user's folders:", err)
	}
	app.Logger.Info.Println("User's folders has been created")

	// create GPT threads for users
	app.Logger.Info.Println("Creating the user's GPT threads")
	gptThreadIDs, err := app.createUserGPTThreads()
	if err != nil {
		app.Logger.Error.Println("Error creating user's GPT threads:", err)
	}
	app.Logger.Info.Println("User's GPT threads have been created")

	// Store user's data in database
	app.Logger.Info.Println("Add the user's data to database")
	userData, err := app.FirestoreClient.CreateUserData(userFolders, gptThreadIDs)
	if err != nil {
		app.Logger.Error.Println("Error creating new user's data:", err)
	}
	app.Logger.Info.Println("User's data has been added")
	return userData
}

func (app *App) createUserGPTThreads() (*db.GPTThreadIDs, error) {
	userGPTThreads := db.GPTThreadIDs{}
	threadIDAddrs := [3]*string{&userGPTThreads.Strategy}
	resultChannel, errChannel := make(chan string), make(chan error)

	// create gpt threads concurrently
	for i := 0; i < len(threadIDAddrs); i++ {
		go func() {
			thread, err := app.GPTClient.CreateThread()
			if err != nil {
				app.Logger.Error.Println("Error creating GPT thread:", err)
				errChannel <- err
				return
			}
			resultChannel <- thread.ID
		}()
	}

	// check the result of each thread creation and update the thread ID
	for i := range threadIDAddrs {
		select {
		case res := <-resultChannel:
			*threadIDAddrs[i] = res
		case err := <-errChannel:
			app.Logger.Error.Println("Error creating GPT thread:", err)
			return nil, err
		}
	}

	// return the user's GPT threads
	return &userGPTThreads, nil
}

func (app *App) createUserIfNotExist(userID string) *db.UserData {
	user, err := app.FirestoreClient.GetUserData(userID)
	if err != nil {
		app.Logger.Warn.Println("User not found, creating new user...")
		userData := app.createUser(userID)
		user = userData

		app.Logger.Info.Println("New user created successfully.")
	}

	return user
}

func (app *App) createUserSessionIfNotExist(userID string) *db.UserSession {
	session, err := app.FirestoreClient.GetUserSession(userID)
	if err != nil {
		app.Logger.Warn.Println("User session not found, creating new session")
		session, err = app.FirestoreClient.CreateUserSession(userID)
		if err != nil {
			app.Logger.Error.Println("Error creating new user session:", err)
		}
	}

	return session
}

func (app *App) getUserPortfolio(user *db.UserData, skill string) *map[string]db.Work {
	var work map[string]db.Work
	switch skill {
	case "clear":
		work = user.Portfolio.Clear
	case "drop":
		work = user.Portfolio.Drop
	case "footwork":
		work = user.Portfolio.Footwork
	case "lift":
		work = user.Portfolio.Lift
	case "netplay":
		work = user.Portfolio.Netplay
	case "strategy":
		work = user.Portfolio.Strategy
	}
	return &work
}
