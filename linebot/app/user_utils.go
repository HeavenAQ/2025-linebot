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
	userFolders, err := app.StorageClient.CreateUserFolders(userID, username)
	if err != nil {
		app.Logger.Error.Println("Error creating new user's folders:", err)
	}
	app.Logger.Info.Println("User's folders has been created")

    // create GPT conversations for users
    app.Logger.Info.Println("Creating the user's GPT conversations")
    gptConversationIDs, err := app.createUserGPTConversations()
    if err != nil {
        app.Logger.Error.Println("Error creating user's GPT conversations:", err)
    }
    app.Logger.Info.Println("User's GPT conversations have been created")

	// Store user's data in database
	app.Logger.Info.Println("Add the user's data to database")
    userData, err := app.FirestoreClient.CreateUserData(userFolders, gptConversationIDs)
	if err != nil {
		app.Logger.Error.Println("Error creating new user's data:", err)
	}
	app.Logger.Info.Println("User's data has been added")
	return userData
}

func (app *App) createUserGPTConversations() (*db.GPTConversationIDs, error) {
    userGPTConversations := db.GPTConversationIDs{}
    idAddrs := [3]*string{&userGPTConversations.Serve, &userGPTConversations.Clear, &userGPTConversations.Smash}
    resultChannel, errChannel := make(chan string), make(chan error)

    // create gpt conversations concurrently
    for i := 0; i < len(idAddrs); i++ {
        go func() {
            conv, err := app.GPTClient.CreateConversation()
            if err != nil {
                app.Logger.Error.Println("Error creating GPT conversation:", err)
                errChannel <- err
                return
            }
            resultChannel <- conv.ID
        }()
    }

    // check the result of each conversation creation and update the ID
    for i := range idAddrs {
        select {
        case res := <-resultChannel:
            *idAddrs[i] = res
        case err := <-errChannel:
            app.Logger.Error.Println("Error creating GPT conversation:", err)
            return nil, err
        }
    }

    // return the user's GPT conversations
    return &userGPTConversations, nil
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
	case "serve":
		work = user.Portfolio.Serve
	case "smash":
		work = user.Portfolio.Smash
	case "clear":
		work = user.Portfolio.Clear
	}
	return &work
}

func (app *App) getUserGPTConversation(user *db.UserData, skill string) string {
    switch skill {
    case "serve":
        return user.GPTConversationIDs.Serve
    case "smash":
        return user.GPTConversationIDs.Smash
    case "clear":
        return user.GPTConversationIDs.Clear
    default:
        return ""
    }
}
