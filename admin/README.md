# BudSimulator Admin Panel

A comprehensive administration panel for the BudSimulator platform, providing analytics, CRUD operations, and user feedback management.

## Features

- **User Analytics**: Track total users, active users, geographical distribution
- **Model Management**: CRUD operations for AI models
- **Hardware Management**: CRUD operations for hardware configurations
- **Use Case Management**: CRUD operations for industry use cases
- **Feedback System**: Manage and respond to user feedback
- **Audit Logging**: Track all administrative actions
- **Real-time Analytics**: Dashboard with charts and metrics

## Setup

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd admin/backend
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file from the example:
   ```bash
   cp .env.example .env
   ```

4. Edit `.env` and update the settings:
   - Change `SECRET_KEY` to a secure random string
   - Update `ADMIN_USERNAME` and `ADMIN_PASSWORD`
   - Ensure `DATABASE_URL` points to the correct database

5. Run the backend:
   ```bash
   python run.py
   ```

   The API will be available at `http://localhost:8001`
   API documentation at `http://localhost:8001/docs`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd admin/frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

   The frontend will be available at `http://localhost:3001`

## Default Credentials

- Username: `admin`
- Password: `changeme123`

**Important**: Change these credentials immediately after first login!

## Architecture

### Backend (FastAPI)
- **Authentication**: JWT-based authentication
- **Database**: Uses existing BudSimulator SQLite database + admin tables
- **Analytics**: Tracks API usage, user sessions, and system metrics
- **CRUD Operations**: Proxies to main API for models/hardware/usecases
- **Audit Logging**: Tracks all administrative actions

### Frontend (React + TypeScript)
- **UI Framework**: Tailwind CSS
- **Charts**: Recharts for data visualization
- **State Management**: React Context for authentication
- **API Client**: Axios with interceptors for auth

## Security Considerations

1. **Change Default Credentials**: Update admin username/password immediately
2. **Use HTTPS**: Deploy with SSL certificates in production
3. **Environment Variables**: Never commit `.env` files
4. **CORS**: Configure allowed origins for production
5. **Database Backups**: Regular backups of the database

## Development

### Adding New Features

1. **Backend**: Add new routes in `api/` directory
2. **Frontend**: Add new pages in `pages/` directory
3. **Database**: Add new models in `models/analytics.py`
4. **Services**: Add new services in `frontend/src/services/`

### Running Tests

Backend:
```bash
cd admin/backend
pytest
```

Frontend:
```bash
cd admin/frontend
npm test
```

## Deployment

### Production Backend

1. Use a production WSGI server:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
   ```

2. Set environment variables for production
3. Use a proper database (PostgreSQL recommended)
4. Enable HTTPS with reverse proxy (nginx)

### Production Frontend

1. Build the frontend:
   ```bash
   npm run build
   ```

2. Serve the build directory with a web server
3. Configure API URL in environment variables

## Monitoring

The admin panel tracks its own metrics:
- API response times
- Error rates
- Active sessions
- System resource usage

Access these metrics in the Dashboard page.

## Support

For issues or questions:
1. Check the main BudSimulator documentation
2. Review API documentation at `/docs`
3. Check audit logs for debugging