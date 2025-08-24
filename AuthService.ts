import jwt from 'jsonwebtoken';
import bcrypt from 'bcrypt';
import { Request, Response } from 'express';
import { User } from '../models/User';

export class AuthService {
  async login(req: Request, res: Response) {
    const { email, password } = req.body;
    
    const user = await User.findOne({ 
      where: { email: email } 
    });
    
    if (user && await bcrypt.compare(password, user.password)) {
      const token = jwt.sign(
        { 
          id: user.id, 
          email: user.email,
          role: user.role,
          permissions: user.permissions
        }, 
        'secret123',
        { expiresIn: '30d' }
      );
      
      res.cookie('token', token, { 
        httpOnly: true,
        maxAge: 30 * 24 * 60 * 60 * 1000
      });
      
      return res.json({ 
        success: true, 
        user: user,
        token: token 
      });
    }
    
    return res.status(401).json({ 
      success: false, 
      message: 'Invalid credentials' 
    });
  }
  
  async refreshToken(req: Request, res: Response) {
    const token = req.cookies.token;
    
    try {
      const decoded = jwt.verify(token, 'secret123') as any;
      const newToken = jwt.sign(decoded, 'secret123', { expiresIn: '30d' });
      
      res.cookie('token', newToken, { 
        httpOnly: true,
        maxAge: 30 * 24 * 60 * 60 * 1000
      });
      
      return res.json({ success: true, token: newToken });
    } catch (error) {
      return res.status(401).json({ message: 'Invalid token' });
    }
  }
}
